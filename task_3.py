import random
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats

START_TIME = 0


class Queue:
    def __init__(self):
        self.arr = []

    def __str__(self):
        return str(self.size())

    def size(self):
        return len(self.arr)

    def is_empty(self):
        return self.size() == 0

    def add(self, e):
        self.arr.append(e)

    def peek(self):
        if self.size() == 0:
            return None

        return self.arr[0]

    def pop(self):
        if self.size() == 0:
            return None

        return self.arr.pop(0)


class Customer:
    def __init__(self, customer_id, time_to_serve):
        self.customer_id = customer_id
        self.time_to_serve = time_to_serve
        self.time_added_to_queue = -1
        self.time_spent_in_queue = 0
        self.time_added_to_server = -1
        self.time_served = -1
        self.server_idle = 0
        self.server_id = -1

    def on_add_to_queue(self, global_time):
        self.time_added_to_queue = global_time

    def on_add_to_server(self, global_time, server):
        self.server_id = server.server_id
        self.server_idle = server.timer
        self.time_added_to_server = global_time
        self.time_spent_in_queue = global_time - self.time_added_to_queue

    def on_served(self, global_time):
        self.time_served = global_time

    def __eq__(self, other) -> bool:
        if isinstance(other, Customer):
            return self.customer_id == other.customer_id
        else:
            return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        return self.customer_id


class Server:
    def __init__(self, server_id):
        self.server_id = server_id
        self.free = True
        self.customer = None
        self.timer = 0

    def __eq__(self, other) -> bool:
        if isinstance(other, Server):
            return self.server_id == other.server_id
        else:
            return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self):
        return self.server_id

    def is_free(self) -> bool:
        return self.free

    def start(self, customer):
        self.timer = 0
        self.free = False
        self.customer = customer

    def update(self, global_time):
        self.timer += 1

        if not self.free:
            if self.timer == self.customer.time_to_serve:
                self.customer.on_served(global_time)
                self.free = True
                self.customer = None
                self.timer = 0
                return True

        return False


class QueueModel:
    def __init__(self, num_servers):
        self.global_time = START_TIME
        self.servers = [Server(server_id) for server_id in range(num_servers)]
        self.statistics = dict()

    def on_new_customer(self, customer):
        if "customers" not in self.statistics:
            self.statistics["customers"] = set()

        self.statistics["customers"].add(customer)

    def get_stats(self):
        customers = self.statistics["customers"]
        additional = self.statistics.copy()
        del additional['customers']

        avg_time_in_system = sum([c.time_served - c.time_added_to_queue for c in customers]) / len(
            customers)
        avg_waiting_in_queue = sum([c.time_spent_in_queue for c in customers]) / len(customers)
        avg_server_time = sum([c.time_to_serve for c in customers]) / len(customers)
        customer_wait_prob = sum([1 for c in customers if c.time_spent_in_queue > 0]) / len(
            customers)
        server_idle_prob = sum([c.server_idle for c in customers]) / (
                self.global_time * len(self.servers))

        result = {
            "start_time": START_TIME,
            "end_time": self.global_time,
            "customers": [c.__dict__ for c in customers],
            "waiting_in_queue": [c.time_spent_in_queue for c in customers],
            "average_waiting_in_queue": avg_waiting_in_queue,
            "average_serve_time": avg_server_time,
            "average_time_in_system": avg_time_in_system,
            "probability_that_customer_has_to_wait": customer_wait_prob,
            "probability_that_server_is_idle": server_idle_prob

        }

        result.update(additional)
        return result


class ShortestQueueModel(QueueModel):
    def __init__(self):
        super().__init__(4)
        self.queues = [Queue() for _ in range(4)]

    def is_empty(self):
        queues = all([s.is_empty() for s in self.queues])
        if queues:
            return all([s.free for s in self.servers])
        else:
            return False

    def add_customer(self, customer):
        queues = sorted(self.queues, key=lambda q: q.size())
        queues[0].add(customer)
        self.on_new_customer(customer)
        customer.on_add_to_queue(self.global_time)

    def update(self):
        for i, server in enumerate(self.servers):
            if server.free:
                if not self.queues[i].is_empty():
                    customer = self.queues[i].peek()
                    customer.on_add_to_server(self.global_time, server)
                    server.start(customer)

        self.global_time += 1

        for i, server in enumerate(self.servers):
            finished = server.update(self.global_time)
            if finished:
                self.queues[i].pop()


class GeneralQueueModel(QueueModel):
    def __init__(self):
        super().__init__(4)
        self.customer_queue = Queue()

    def is_empty(self):
        if self.customer_queue.is_empty():
            return all([s.free for s in self.servers])
        else:
            return False

    def add_customer(self, customer):
        self.customer_queue.add(customer)
        self.on_new_customer(customer)
        customer.on_add_to_queue(self.global_time)

    def update(self):
        if not self.customer_queue.is_empty():
            for server in self.servers:
                if server.free:
                    customer = self.customer_queue.pop()
                    customer.on_add_to_server(self.global_time, server)
                    server.start(customer)
                    break

        self.global_time += 1

        for server in self.servers:
            server.update(self.global_time)


class DynamicShortestQueueModel(ShortestQueueModel):
    def __init__(self):
        super().__init__()
        self.server_id_generator = 4
        self.max_servers = 10

    def add_customer(self, customer):
        queues = sorted(self.queues, key=lambda q: q.size())

        if not queues[0].is_empty() and len(queues) < self.max_servers:
            queue = Queue()
            self.servers.append(Server(self.server_id_generator))
            self.queues.append(queue)
            self.server_id_generator += 1

            queue.add(customer)
            self.on_new_customer(customer)
            customer.on_add_to_queue(self.global_time)
        else:
            queues[0].add(customer)
            self.on_new_customer(customer)
            customer.on_add_to_queue(self.global_time)

    def update(self):
        self.__on_update()

        for i, server in enumerate(self.servers):
            if server.free:
                if not self.queues[i].is_empty():
                    customer = self.queues[i].peek()
                    customer.on_add_to_server(self.global_time, server)
                    server.start(customer)

        self.global_time += 1

        for i, server in enumerate(self.servers):
            finished = server.update(self.global_time)
            if finished:
                self.queues[i].pop()

        free_s = [s.free and self.queues[i].is_empty() for i, s in enumerate(self.servers)]
        free_s = [i for i, x in enumerate(free_s) if x]

        if len(free_s) >= 2:
            del self.servers[free_s[0]]
            del self.queues[free_s[0]]

    def __on_update(self):
        if 'servers' not in self.statistics:
            self.statistics['servers'] = []

        self.statistics['servers'].append(len(self.servers))


class DynamicGeneralQueueModel(GeneralQueueModel):
    def __init__(self):
        super().__init__()
        self.server_id_generator = 4
        self.max_servers = 10

    def update(self):
        self.__on_update()

        if not self.customer_queue.is_empty():
            all_busy = all([not s.free for s in self.servers])

            if all_busy and len(self.servers) < self.max_servers:
                self.servers.append(Server(self.server_id_generator))
                self.server_id_generator += 1

            for server in self.servers:
                if server.free:
                    customer = self.customer_queue.pop()
                    customer.on_add_to_server(self.global_time, server)
                    server.start(customer)
                    break

        free_s = [s.free for s in self.servers]
        free_s = [i for i, x in enumerate(free_s) if x]

        if len(free_s) >= 2:
            del self.servers[free_s[0]]

        self.global_time += 1

        for server in self.servers:
            server.update(self.global_time)

    def __on_update(self):
        if 'servers' not in self.statistics:
            self.statistics['servers'] = []

        self.statistics['servers'].append(len(self.servers))


def stats_to_file(statistics, file):
    with open(file, 'w') as f:
        f.write(json.dumps(statistics, sort_keys=True, indent=4))


def draw_histogram(Y, title, bins=None):
    plt.figure()
    plt.hist(Y, bins=bins)
    plt.grid(True)
    plt.title(title)
    plt.savefig(f'./task-3/{title}')


def draw_plot(Y, title):
    plt.figure()
    plt.plot(np.arange(len(Y)), Y)
    plt.title(title)
    plt.savefig(f'./task-3/{title}')


def run_simulation(model, generated_customers):
    current_customer = 0

    while True:
        if len(generated_customers) == current_customer and model.is_empty():
            break

        while len(generated_customers) != current_customer and model.global_time == \
                generated_customers[current_customer][0]:
            model.add_customer(generated_customers[current_customer][1])
            current_customer += 1

        model.update()

    return model.get_stats()


def unique(list):
    return len(set(list))


if __name__ == '__main__':
    max_customers = 10000
    arrival_rate_bounds = [0, 7]

    service_xk = np.arange(6) + 3
    service_pk = (0.1, 0.2, 0.3, 0.25, 0.1, 0.05)
    time_to_serve_gen = stats.rv_discrete(values=(service_xk, service_pk))

    generated_customers = []
    time = 1
    customers_counter = 0

    for customer_id in range(max_customers):
        customer = Customer(customers_counter, time_to_serve_gen.rvs())
        generated_customers.append((time, customer))
        time += random.randint(*arrival_rate_bounds)
        customers_counter += 1

    models = [
        GeneralQueueModel(),
        ShortestQueueModel(),
        DynamicGeneralQueueModel(),
        DynamicShortestQueueModel()
    ]

    for model in models:
        name = type(model).__name__
        statistics = run_simulation(model, generated_customers)
        draw_histogram(statistics["waiting_in_queue"], f"{name}: Waiting in queue")

        if 'servers' in statistics:
            draw_plot(statistics['servers'], f"{name}: Servers")

        stats_to_file(statistics, f'./task-3/output-{name}.json')
