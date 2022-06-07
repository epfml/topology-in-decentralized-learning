from typing import Dict, Tuple

import torch
import random
import torch.distributed as dist
from topologies import Topology


def is_dist_avail_and_initialized():
    """Copied from https://github.com/facebookresearch/deit/blob/main/utils.py"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Copied from https://github.com/facebookresearch/deit/blob/main/utils.py"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Copied from https://github.com/facebookresearch/deit/blob/main/utils.py"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def isend(tensor, *args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # print("send [{}] {}->{}, shape {}, value {}".format(kwargs["tag"], get_rank(), args[0], tensor.shape, tensor.view(-1)[0].item()))
    return torch.distributed.isend(tensor, *args, **kwargs)


def recv(*args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return torch.distributed.recv(*args, **kwargs)


def irecv(*args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return torch.distributed.irecv(*args, **kwargs)


def pack(tensors):
    """Packs a list of tensors into one buffer for sending to other workers"""
    buffer = torch.cat([t.view(-1) for t in tensors])  # copies
    shapes = [tensor.shape for tensor in tensors]
    return buffer, shapes


def unpack(buffer, shapes):
    """Provides pointers to tensors of original `shapes` in a flat-packed buffer."""
    idx = 0
    entries = []
    for tensor_shape in shapes:
        end = idx + tensor_shape.numel()
        entries.append(buffer[idx:end].view(size=tensor_shape))
        idx = end

    return entries


def num_bytes(tensor):
    return tensor.nelement() * tensor.element_size()


class MultiHandle:
    def __init__(self, handles):
        self.handles = handles

    def wait(self):
        for handle in self.handles:
            handle.wait()
        if torch.cuda.is_available():
            torch.cuda.synchronize()


class RelayMechanism:
    def __init__(
        self,
        topology: Topology,
        overlap: bool = False,
        normalization_mode: str = "world_size",
        initial_state=None,
        message_drop_prob=0,
        tag: int = 2,
    ):
        self._topology = topology
        self._initial_state = initial_state
        self._received_messages = {}
        self._received_counts = {}
        self._tag = tag
        self._overlap = overlap
        self._normalization_mode = normalization_mode
        self._messages_per_worker = topology.max_degree
        self._message_drop_prob = message_drop_prob

        self._send_handle = None
        self._last_sent_tensor = None
        self.bytes_sent = 0

    def send(self, tensor: torch.Tensor, verbose=False):
        """Send `tensor` to neighbors, relaying previously received messages"""
        if self._overlap and self._last_sent_tensor is not None:
            self._received_counts = recv_messages(
                example_message=torch.tensor(1), topology=self._topology, tag=self._tag
            )
            self._received_messages = recv_messages(
                example_message=self._last_sent_tensor,
                topology=self._topology,
                tag=self._tag,
            )

            # Randomly delete some messages for simulations of robustness
            for key in list(self._received_counts.keys()):
                if random.uniform(0, 1) < self._message_drop_prob:
                    del self._received_counts[key]
                    del self._received_messages[key]

            self._send_handle.wait()

        count_handle, delta_bytes_1 = isend_to_neighbors(
            torch.tensor(1), self._topology, relay=self._received_counts, tag=self._tag
        )

        if verbose:
            print("    sending {:.5f} ({})".format(tensor[0], self._tag))

        msg_handle, delta_bytes_2 = isend_to_neighbors(
            tensor, self._topology, relay=self._received_messages, tag=self._tag
        )
        self._send_handle = MultiHandle([count_handle, msg_handle])
        self._last_sent_tensor = tensor

        self.bytes_sent += (delta_bytes_1 + delta_bytes_2) * self._messages_per_worker / self._topology.max_degree

    def set_initial_state(self, tensor: torch.Tensor):
        """Set the fallback for when not enough messages arrive"""
        self._initial_state.data = tensor.clone()

    def receive_average(self, verbose=False):
        """Form an average from received message and the last sent tensor"""
        assert self._last_sent_tensor is not None

        count_tensor = torch.tensor(1)
        avg_tensor = self._last_sent_tensor.clone()

        if verbose:
            print("      start with {:.5f}".format(avg_tensor[0]))

        if not self._overlap:
            self._received_counts = recv_sum_into_tensor(
                tensor=count_tensor, topology=self._topology, tag=self._tag
            )
            self._received_messages = recv_sum_into_tensor(
                tensor=avg_tensor, topology=self._topology, tag=self._tag
            )

            if verbose:
                print("      received_counts {:.5f}".format(count_tensor.item()))
                print("      sum {:.5f}".format(avg_tensor[0]))
                for key, msg in self._received_messages.items():
                    print("      msg {} {:.5f}".format(key, msg[0]))

            # Randomly delete some messages for simulations of robustness
            for key in list(self._received_counts.keys()):
                if random.uniform(0, 1) < self._message_drop_prob:
                    count_tensor.sub_(self._received_counts[key])
                    del self._received_counts[key]
                    avg_tensor.sub_(self._received_messages[key])
                    del self._received_messages[key]

            self._send_handle.wait()
        else:
            for count in self._received_counts.values():
                count_tensor.add_(count)
            for message in self._received_messages.values():
                avg_tensor.add_(message)

        if self._normalization_mode == "world_size":
            num_workers = self._topology.num_workers

            # Supplement missing values by 'initial state'.
            # This will be 0 for RelaySum/Grad and x0 for RelaySum/Model.
            if count_tensor < get_world_size() and self._initial_state is not None:
                avg_tensor.add_(self._initial_state, alpha=num_workers - count_tensor)

            avg_tensor.div_(num_workers)

        elif self._normalization_mode == "counts":
            avg_tensor.div_(count_tensor)

        else:
            raise ValueError(f"Unknown normalization mode {self._normalization_mode}")

        return avg_tensor


class MultiTopologyRelayMechanism:
    """
    Divide communication evenly over multiple topologies
    """
    def __init__(self, topologies, **kwargs):
        if not isinstance(topologies, list):
            topologies = [topologies]
        
        initial_state = kwargs.pop("initial_state", None)
        if initial_state is None:
            initial_states = [None for _ in topologies]
        else:
            initial_states = [initial_state[i::len(topologies)] for i in range(len(topologies))]

        self._relays = [RelayMechanism(t, initial_state=s, tag=10+i, **kwargs) for i, (t, s) in enumerate(zip(topologies, initial_states))]

        # Make sure the number of bytes sent is tracked in a representive way.
        messages_per_worker = 0
        for worker in range(topologies[0].num_workers):
            avg_num_neighbors = sum(len(topo.neighbors(worker)) for topo in topologies) / len(topologies)
            messages_per_worker = max(avg_num_neighbors, messages_per_worker)
        print(f"We'll count {messages_per_worker} per worker.")
        for r in self._relays:
            r._messages_per_worker = messages_per_worker

    def send(self, tensor: torch.Tensor):
        if len(self._relays) == 1:
            return self._relays[0].send(tensor)
        else:
            self._last_sent = tensor
            for i, relay in enumerate(self._relays):
                relay.send(tensor[i::len(self._relays)], verbose=False)

    def set_initial_state(self, tensor: torch.Tensor):
        if len(self._relays) == 1:
            return self._relays[0].set_initial_state(tensor)
        else:
            self._last_sent = tensor
            for i, relay in enumerate(self._relays):
                relay.set_initial_state(tensor[i::len(self._relays)])

    def receive_average(self):
        if len(self._relays) == 1:
            return self._relays[0].receive_average()
        else:
            avg_tensor = torch.zeros_like(self._last_sent)
            for i, relay in enumerate(self._relays):
                avg_tensor[i::len(self._relays)] = relay.receive_average(verbose=False)
            return avg_tensor

    @property
    def bytes_sent(self):
        return sum(r.bytes_sent for r in self._relays)


class GossipMechanism:
    def __init__(
        self,
        topology: Topology,
        gossip_matrix = None,
        message_drop_prob=0,
        tag: int = 2,
    ):
        self._topology = topology
        self._w = gossip_matrix if gossip_matrix is not None else topology.gossip_matrix()
        self._message_drop_prob = message_drop_prob
        self._tag = tag
        self.bytes_sent = 0

    def send(self, tensor: torch.Tensor):
        self._send_handle, delta_bytes = isend_to_neighbors(
            tensor, self._topology, tag=self._tag
        )
        self._last_sent_tensor = tensor.clone()
        self.bytes_sent += delta_bytes

    def gossip_update(self, tensor: torch.Tensor):
        """Changes tensor in-place"""
        assert self._last_sent_tensor is not None
        rank = get_rank()
        tensor.sub_(self._last_sent_tensor, alpha = 1-self._w[rank, rank])

        total_weight = 1-self._w[rank, rank]

        recv_buffer = torch.empty_like(tensor)
        for neighbor in self._topology.neighbors(rank):
            recv(recv_buffer, neighbor, tag=self._tag)
            if self._message_drop_prob == 0 or random.uniform(0, 1) > self._message_drop_prob:
                tensor.add_(recv_buffer, alpha=self._w[rank, neighbor])
                total_weight += self._w[rank, neighbor]

        # If not all messages arrived, we should re-normalize
        if total_weight < 1:
            tensor.mul_(1/total_weight)

        self._send_handle.wait()



class MultiTopologyGossipMechanism:
    """
    Divide communication evenly over multiple topologies
    """
    def __init__(self, topologies, **kwargs):
        if not isinstance(topologies, list):
            topologies = [topologies]

        self._gossips = [GossipMechanism(t, **kwargs) for t in topologies]

    def send(self, tensor: torch.Tensor):
        if len(self._gossips) == 1:
            return self._gossips[0].send(tensor)
        else:
            for i, gossip in enumerate(self._gossips):
                gossip.send(tensor[i::len(self._gossips)])

    def gossip_update(self, tensor: torch.Tensor):
        """Changes tensor in-place"""
        if len(self._gossips) == 1:
            return self._gossips[0].gossip_update(tensor)
        else:
            for i, gossip in enumerate(self._gossips):
                gossip.gossip_update(tensor[i::len(self._gossips)])

    @property
    def bytes_sent(self):
        return sum(g.bytes_sent for g in self._gossips)



def isend_to_neighbors(
    buffer, topology: Topology, relay: Dict[int, torch.Tensor] = {}, tag=2
):
    handles = []
    neighbors = topology.neighbors(get_rank())
    for neighbor in neighbors:
        total_relay = sum(
            msg for msg_origin, msg in relay.items() if msg_origin != neighbor
        )
        handle = isend(buffer + total_relay, neighbor, tag=tag)
        handles.append(handle)

    bytes_sent = num_bytes(buffer) * topology.max_degree  # worst case

    return MultiHandle(handles), bytes_sent


def irecv_messages(
    example_message: torch.Tensor, topology: Topology, tag=2
) -> Dict[int, torch.Tensor]:
    """Receive one message from each neighbor"""
    received_messages = {}

    handles = []

    for neighbor in topology.neighbors(get_rank()):
        received_messages[neighbor] = torch.zeros_like(example_message)
        handle = irecv(received_messages[neighbor], neighbor, tag=tag)
        handles.append(handle)

    return received_messages, MultiHandle(handles)


def recv_messages(
    example_message: torch.Tensor, topology: Topology, tag=2
) -> Dict[int, torch.Tensor]:
    """Receive one message from each neighbor"""
    received_messages, handle = irecv_messages(example_message, topology, tag)
    handle.wait()

    # for sender, msg in received_messages.items():
    #     print("recv [{}] {}->{}, shape {}, value {}".format(tag, sender, get_rank(), msg.shape, msg.view(-1)[0].item()))

    return received_messages


def recv_sum_into_tensor(
    tensor: torch.Tensor, topology: Topology, tag=2
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Add a message from all neighbors to `tensor` (in place)"""
    received_messages = recv_messages(tensor, topology, tag=tag)

    for msg in received_messages.values():
        tensor.add_(msg)

    return received_messages
