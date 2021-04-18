import random
import threading

from flask import Flask, request, redirect, send_file

from super_res import Upscaler

app = Flask(__name__, static_folder='results')


@app.route('/')
def index():
    return ''' 
    <form action="/upscale" method="POST" enctype="multipart/form-data">
        <label>file</label>
        <input type="file" name="file" id="file"  />
        <select name="model" id="model">
            <option value="">--Please choose an option--</option>
            <option value="danbooru">Danbooru</option>
            <option value="painting">Painting</option>
        </select>
        <input type="submit" value="Upscale!">
    </form>
    '''

status = {}
max_resource = 2
current_resource = 0


@app.route('/result/<id>')
def get_result(id):
    return send_file(f'results/{id}.png')

def do_upscale(upscaler: Upscaler, id: int):
    print('upscaling')
    global status
    for idx in range(len(upscaler)):
        status[id] = f'upscaling tile {idx}/{len(upscaler)}'
        print('upscaling tile ', idx, '/', len(upscaler))
        upscaler.upscale_tile(idx)
    upscaler.result.save(f'results/{id}.png')
    global current_resource
    current_resource = current_resource - 1

@app.route('/status/<id>')
def get_status(id):
    global status
    print(status)
    return status[int(id)] + f'<img src="/result/{id}" />'


@app.route('/upscale', methods=["POST"])
def upscale():
    global current_resource
    global max_resource
    print(f'{current_resource} / {max_resource}')
    if current_resource >= max_resource:
        return 'too many requests. please try again later.'
    print(request.form)
    models = {
        'danbooru':  'C:\\super_res_danbooru_resnet50.onnx',
        'painting': 'C:\\super_res_painting_resnet50.onnx'
    }
    upscaler = Upscaler(request.files['file'], models[request.form['model']], scale=4, grid_size=64,
                        overlap_factor=2, device='cuda')
    id = random.randint(0, 9999)
    upscaler.upscale_tile(0)
    x = threading.Thread(target=do_upscale, args=(upscaler, id))
    x.start()
    current_resource = current_resource + 1
    return redirect('/status/' + str(id))


app.run(host='0.0.0.0')