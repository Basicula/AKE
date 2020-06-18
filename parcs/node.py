import imp
from threading import Thread

import time

import Pyro4
import json
from abc import abstractmethod
from .node_info import get_node_info_for_current_machine
from .node_link import NodeLink
from .file_utils import get_solution_path, setup_working_directory
import requests
import logging

log = None

class Node:
    def __init__(self, conf, scenes_root = "./parcsv2/Scenes"):
        self.conf = conf
        self.info = get_node_info_for_current_machine()
        self.scenes_root = scenes_root

    @abstractmethod
    def is_master_node(self):
        pass

    @staticmethod
    def create_node(conf, scenes_root = "./parcsv2/Scenes"):
        if conf.master:
            return MasterNode(conf, scenes_root)
        else:
            return WorkerNode(conf, scenes_root)


class WorkerNode(Node):
    def __init__(self, conf, scenes_root = "./parcsv2/Scenes"):
        Node.__init__(self, conf, scenes_root)
        global log
        log = logging.getLogger('Worker Node')
        self.master = NodeLink(conf.master_ip, conf.master_port)
        self.connected = False
        log.info('Started on %s:%d;', conf.ip, conf.port)
        self.register_on_master()
        #self.reconnector = MasterReconnector(self)
        #self.reconnector.start()
    
    def is_master_node(self):
        return False

    def register_on_master(self):
        data = {'ip': self.conf.ip, 'port': self.conf.port,
                'info': {'cpu': self.info.cpu, 'ram': self.info.ram}}
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        try:
            r = requests.post('http://%s:%s/api/internal/worker' % (self.master.ip, self.master.port),
                              data=json.dumps(data), headers=headers)
            if r.status_code == 200:
                self.connected = True
                log.info('Registered to master on %s:%d.',
                         self.conf.master_ip, self.conf.master_port)
            else:
                self.connected = False
                log.warning('Unable to register to master on %s:%d.', self.conf.master_ip, self.conf.master_port)
        except Exception as e:
            self.connected = False
            log.warning('Unable to register to master on %s:%d because of %s.', self.conf.master_ip,
                        self.conf.master_port, str(e))

    def connection_with_master_lost(self):
        self.connected = False
        log.warning('Connection with master %s:%d lost.', self.master.ip, self.master.port)
        


class MasterReconnector(Thread):
    def __init__(self, worker_node):
        super(MasterReconnector, self).__init__()
        self.setDaemon(True)
        self.worker_node = worker_node

    def run(self):
        while True:
            try:
                time.sleep(1)
                log.info('Trying to connect')
                response = requests.get('http://%s:%s/api/internal/heartbeat' % (
                    self.worker_node.conf.ip, self.worker_node.conf.port))
                if response.status_code == 200:
                    break
            except Exception as e:
                print(e)
                pass
        while True:
            if self.worker_node.connected:
                try:
                    response = requests.get('http://%s:%s/api/internal/heartbeat' % (
                        self.worker_node.master.ip, self.worker_node.master.port))
                    if response.status_code != 200:
                        self.worker_node.connection_with_master_lost()
                except Exception as e:
                    self.worker_node.connection_with_master_lost()
            else:
                self.worker_node.register_on_master()
            time.sleep(5)


class MasterNode(Node):
    def __init__(self, conf, scenes_root = "./parcsv2/Scenes"):
        Node.__init__(self, conf, scenes_root)
        global log
        log = logging.getLogger('Master Node')
        self.workers = []
        log.info('Started on %s:%d;', conf.ip, conf.port)
        self.heartbeat = Heartbeat(self)
        self.heartbeat.start()

    def is_master_node(self):
        return True

    def register_worker(self, node_link):
        same_links = list(filter(lambda l: l.ip == node_link.ip and l.port == node_link.port, self.workers))
        if len(same_links) == 0:
            self.workers.append(node_link)
            ret = True
        else:
            log.warning('Unable to register node %s:%d because it is already registered.', node_link.ip, node_link.port)
            ret = False
        return ret

    def find_worker(self, worker_id):
        workers_list = list(filter(lambda w: w.id == worker_id, self.workers))
        return None if len(workers_list) == 0 else workers_list[0]

    def delete_worker(self, worker_id):
        prev_len = len(self.workers)
        self.workers = list(filter(lambda w: w.id != worker_id, self.workers))
        return prev_len != len(self.workers)
        

class Heartbeat(Thread):
    def __init__(self, master_node):
        super(Heartbeat, self).__init__()
        self.setDaemon(True)
        self.master_node = master_node
        global log
        log = logging.getLogger('Heartbeat')

    def run(self):
        while True:
            time.sleep(5)
            log.debug('%d workers is about to check.', len(self.master_node.workers))
            dead_workers = []
            for worker in self.master_node.workers:
                try:
                    response = requests.get('http://%s:%s/api/internal/heartbeat' % (worker.ip, worker.port))
                    if response.status_code != 200:
                        dead_workers.append(worker.id)
                except Exception as e:
                    dead_workers.append(worker.id)
            if len(dead_workers) == 0:
                log.debug('All workers alive.')
            else:
                log.warn('%d workers are dead.', len(dead_workers))
            for dead_worker in dead_workers:
                self.master_node.delete_worker(dead_worker)
