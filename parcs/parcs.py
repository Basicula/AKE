import logging
import requests
from requests_futures.sessions import FuturesSession
import json
import os
import random
from flask import Flask, render_template, request, send_from_directory, jsonify, Response, make_response

from .node import Node
from .node_link import create_node_link

from engine import Scene
from engine.Fluid import Fluid
from engine.Common import BoundingBox
from engine.Visual import Image, Color
from engine.Visual.Material import ColorMaterial
from engine.Primitives import Sphere
from engine.Math.Vector import Vector3d

from enum import Enum
class Scenario(Enum):
    CAMERA_PER_WORKER = 1
    PART_OF_SCREEN_PER_WORKER = 2

def start(conf):
    log.info("Starting...")
    app.node = Node.create_node(conf,"./parcsv2/Scenes")
    log.info("Started.")
    app.run(host='0.0.0.0', port=conf.port)
    app.inited = False

logging.basicConfig(level=logging.INFO)

log = logging.getLogger('PARCS')

app = Flask(__name__)
app.debug = False
app.node = None
app.inited = False
app.scenario = Scenario.CAMERA_PER_WORKER

def bad_request():
    return Response(status=400)


def not_found():
    return Response(status=404)


def ok():
    return Response(status=200)

# WEB
@app.route('/')
@app.route('/index')
def index_page():
    scenes_folder = os.walk(app.node.scenes_root)
    scenes = []
    for addr, dirs, files in scenes_folder:
        for file in files:
            if '.json' in file:
                scene = {}
                scene["name"] = file.split('.')[0]
                scene["file"] = file
                scenes.append(scene)
    return render_template('index.html', scenes = scenes)

@app.route('/about')
def about_page():
    return render_template("about.html")

@app.route('/simulation/<filename>', methods=['GET'])
def simulation(filename):
    scenes_folder = os.walk(app.node.scenes_root)
    scenes = []
    for addr, dirs, files in scenes_folder:
        for file in files:
            if filename in file:
                with open(addr+'/'+file,'r') as f:
                    app.node.scene = Scene.fromDict(json.load(f))
    for worker in app.node.workers:
        response = requests.post(('http://%s:%s/api/internal/worker/scene/init/' + filename) % (worker.ip, worker.port))
    return render_template("simulation.html")
    
@app.route('/simulation/update', methods=['GET'])
def simulation_update():
    session = FuturesSession()
    responses = []
    
    result_images = []
    if app.scenario == Scenario.CAMERA_PER_WORKER:
        num_of_res = 0;
        for worker in app.node.workers:
            responses.append(session.get('http://%s:%s/api/internal/worker/scene/update/camera/%d' % (worker.ip, worker.port, num_of_res)))
            num_of_res += 1
        
        for response in responses:
            response_json = response.result().json()
            if not 'image' in response_json:
                continue
            image = {}
            image['image'] = response_json['image']
            image['width'] = response_json['width']
            image['height'] = response_json['height']
            result_images.append(image)
    else:
        num_of_res = 0;
        for worker in app.node.workers:
            responses.append(session.get('http://%s:%s/api/internal/worker/scene/update' % (worker.ip, worker.port)))
            num_of_res += 1
        
        # get result frames from all workers
        image = ""
        for response in responses:
            image += response.result().json()['image']
            image += chr(0) * 3 * app.node.scene.frameWidth
        
        result_images.append({"image":image, "width" : app.node.scene.frameWidth, "height" : app.node.scene.frameHeight + num_of_res})
    
    return {'images' : result_images}

# Inernal api
@app.route('/api/internal/heartbeat')
def heartbeat():
    return ok()

@app.route('/api/internal/worker', methods=['POST'])
def register_worker():
    json = request.get_json()
    node_link = create_node_link(json)
    log.info("Worker %s is about to register.", str(node_link))
    result = app.node.register_worker(node_link)
    if result:
        log.info("Worker %s registered.", str(node_link))
        return jsonify(worker=node_link.serialize())
    else:
        return bad_request()
        
@app.route('/api/internal/worker/scene/init/<filename>', methods = ['POST'])
def scene_init(filename):
    scenes_folder = os.walk(app.node.scenes_root)
    scenes = []
    for addr, dirs, files in scenes_folder:
        for file in files:
            if filename in file:
                with open(addr+'/'+file,'r') as f:
                    app.node.scene = Scene.fromDict(json.load(f))
                    app.node.scene.addObject(Fluid(32))
    app.image = Image(app.node.scene.frameWidth, app.node.scene.frameHeight)
    return ok()
    
@app.route('/api/internal/worker/scene/update', methods = ['GET'])
def scene_update():
    width = app.node.scene.frameWidth
    height = app.node.scene.frameHeight
    app.node.scene.getFrame(app.image)
    return {"image":app.image.rgbDataStr(), "width" : width, "height" : height}

@app.route('/api/internal/worker/scene/update/camera/<camera_id>', methods = ['GET'])
def scene_camera_update(camera_id):
    camera_id = int(camera_id)
    if camera_id >= app.node.scene.camCnt:
        return {}

    width = app.node.scene.frameWidth
    height = app.node.scene.frameHeight
    app.node.scene.activeCamera = camera_id
    app.node.scene.getFrame(app.image)
    app.node.scene.update()
    return {"image":app.image.rgbDataStr(), "width" : width, "height" : height}    