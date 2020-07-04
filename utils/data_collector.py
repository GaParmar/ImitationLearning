import os, sys, pdb
import time
import numpy as np
from PIL import Image
from threading import Thread
from utils.controllers.web_controller import *
from utils.controllers.physical_controller import *
import gym
import gym_donkeycar
from torchvision import transforms

class DataCollector:
    # initialize the environment, web controller
    def __init__(self, sim_path, controller="web", angle_scale=1.0, throttle_scale=1.0):
        env_name      = "donkey-mountain-track-v0"
        host          = "127.0.0.1"
        port          = 9092
        if controller == "web":
            wc_port       = 8887
            wc_mode       = "user"
            wc = LocalWebController(port=wc_port, mode=wc_mode)
            wc_thread = Thread(target=wc.update, args=())
            wc_thread.daemon = True
            wc_thread.start()
            self.wc = wc
        elif controller == "xbox":
            cont = XboxOneJoystickController()
            cont_thread = Thread(target=cont.update, args=())
            cont_thread.daemon = True
            cont_thread.start()
            self.wc = cont
        self.controller = controller
        self.env  = gym.make(env_name, exe_path=sim_path,
                host=host, port=port)
        self.angle_scale = angle_scale
        self.throttle_scale = throttle_scale

    def reset(self):
        _ = self.env.step(np.array([0, 0]))
        _ = self.env.reset()
    
    def set_policy(self, policy):
        self.policy = policy
    
    def t_throttle(self, t):
        t = self.throttle_scale*t
        t = np.clip(t, 0.0, 1.0)
        return t
    
    def t_angle(self, a):
        a = self.angle_scale*a
        return a

    def collect_data(self, output_folder=None, override=True, buffer_length=100, duration=60, refresh_rate=10, use_policy=False):
        ctr = 0
        buffer = []
        if output_folder is None:
            output_folder = f"output/exp_{time.time()}"
        if override:
            os.system(f"rm -r {output_folder}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # busy wait till recording is started
        while True: 
            if self.wc.run_threaded(img_arr=None)[3] : break
        print("started recording")

        rec_start_ts = time.time()
        obs, reward, done, info = self.env.step(np.array([0, 0]))
        while time.time() < (rec_start_ts+duration):
            start_time = time.time()
            img = Image.fromarray(obs)
            
            if use_policy:
                img = Image.fromarray(obs)
                img_t = transforms.ToTensor()(img).view(1,3,120,160).cuda()
                throttle, angle = self.policy(img_t)
                pred_angle = angle.detach().view(-1).item()
                pred_throttle = throttle.detach().view(-1).item()
                pred_throttle = np.clip(pred_throttle, 0.0, 1.0)
                cont_angle, cont_throttle, _, _ = self.wc.run_threaded(img_arr=img)
            else:
                cont_angle, cont_throttle, _, _ = self.wc.run_threaded(img_arr=img)
    
            if self.controller == "xbox":
                cont_throttle *= -1.0

            cont_throttle = self.t_throttle(cont_throttle)
            cont_angle = self.t_angle(cont_angle)

            if use_policy:
                action = np.array([pred_angle*5.0, pred_throttle])
            else:
                action = np.array([cont_angle*5.0, cont_throttle])
            obs, reward, done, info = self.env.step(action)
            curr = {
                "idx"       : ctr,
                "img"       : img,
                "angle"     : cont_angle,
                "throttle"  : cont_throttle
            }
            # print(abs(info["cte"])>7, done)
            if abs(info["cte"])>7: break
            ctr += 1
            buffer.append(curr)
            if len(buffer) == buffer_length:
                # save the buffer to output folder
                for e in buffer:
                    fname = os.path.join(output_folder, f"{e['idx']}_{e['angle']}_{e['throttle']}.png")
                    e["img"].save(fname)
                buffer = []
            sleep_time = 1.0 / refresh_rate - (time.time() - start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
        print("collection done")


if __name__ == "__main__":
    dc = DataCollector()
    dc.reset()
    dc.collect_data(duration=60)
    dc.reset()