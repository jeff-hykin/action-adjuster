import math
import time
import random
from collections import namedtuple
import json

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
import numpy
import numpy as np
import gym

from __dependencies__.blissful_basics import Csv, create_named_list_class, FS, print, stringify, clip, countdown, LazyDict

class Renderer:
    def __init__(
        self,
        vehicle_render_width,
        vehicle_render_length,
        waypoints_list,
        should_render,
        inital_x,
        inital_y,
        render_axis_size,
        render_path,
        history_size=700,
    ):
        self.should_render = should_render
        self.history_size = history_size
        self.x_pose = [inital_x] * self.history_size
        self.y_pose = [inital_y] * self.history_size
        self.prev_render_timestamp = time.time()
        self.frame_number = -1
        if self.should_render:
            self.vehicle_render_width = vehicle_render_width
            self.vehicle_render_length = vehicle_render_length
            self.warthog_diag   = math.sqrt(vehicle_render_width**2 + vehicle_render_length**2)
            self.diagonal_angle = math.atan2(vehicle_render_length, vehicle_render_width)
            self.waypoints_list = waypoints_list
            self.render_path = render_path
            self.render_axis_size = render_axis_size
            
            print(f'''rendering to: {self.render_path}''')
            FS.remove(self.render_path)
            FS.ensure_is_folder(self.render_path)
            plt.ion
            self.fig = plt.figure(dpi=100, figsize=(10, 10))
            self.ax  = self.fig.add_subplot(111)
            self.ax.set_xlim([-4, 4])
            self.ax.set_ylim([-4, 4])
            self.rect = Rectangle((0.0, 0.0), vehicle_render_width * 2, vehicle_render_length * 2, fill=False)
            self.ax.add_artist(self.rect)
            (self.cur_pos,) = self.ax.plot(self.x_pose, self.y_pose, "+g")
            self.text = self.ax.text(1, 2, "", style="italic", bbox={"facecolor": "red", "alpha": 0.5, "pad": 10}, fontsize=12)
            if type(self.waypoints_list) != type(None):
                self.plot_waypoints()
    
    def render_if_needed(
        self,
        prev_next_waypoint_index,
        x_point, # self.spacial_info.x
        y_point, # self.spacial_info.y
        angle,   # self.spacial_info.angle
        text_data, # f"vel_error={self.velocity_error:.3f}\nclosest_index={self.next_waypoint_index}\ncrosstrack_error={self.crosstrack_error:.3f}\nReward={self.reward:.4f}\nwarthog_vel={self.spacial_info.velocity:.3f}\nphi_error={self.phi_error*180/math.pi:.4f}\nsim step={time.time() - self.prev_timestamp:.4f}\nep_reward={self.total_episode_reward:.4f}\n\nomega_reward={omega_reward:.4f}\nvel_reward={self.velocity_error:.4f}",
        mode="human"
    ):
        self.frame_number += 1
        if self.should_render and self.should_render():
            text_data = f"""sim step={time.time() - self.prev_render_timestamp:.4f}\n{text_data}"""
            self.prev_render_timestamp = time.time()
            # plot all the points in blue
            x = []
            y = []
            for each_x, each_y, *_ in self.waypoints_list:
                x.append(each_x)
                y.append(each_y)
            self.ax.plot(x, y, "+b")
            
            # plot remaining points in red
            x = []
            y = []
            for each_x, each_y, *_ in self.waypoints_list[prev_next_waypoint_index:]:
                x.append(each_x)
                y.append(each_y)
            self.ax.plot(x, y, "+r")
            
            self.ax.set_xlim([x_point - self.render_axis_size / 2.0, x_point + self.render_axis_size / 2.0])
            self.ax.set_ylim([y_point - self.render_axis_size / 2.0, y_point + self.render_axis_size / 2.0])
            total_diag_ang = self.diagonal_angle + angle
            xl = x_point - self.warthog_diag * math.cos(total_diag_ang)
            yl = y_point - self.warthog_diag * math.sin(total_diag_ang)
            self.rect.remove()
            self.rect = Rectangle(
                xy=(xl, yl), 
                width=self.vehicle_render_width * 2, # not sure why * 2 -- Jeff 
                height=self.vehicle_render_length * 2, 
                angle=180.0 * angle / math.pi,
                facecolor="blue",
            )
            self.text.remove()
            self.text = self.ax.text(
                x_point + 1,
                y_point + 2,
                text_data,
                style="italic",
                bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
                fontsize=10,
            )
            self.ax.add_artist(self.rect)
            self.x_pose.append(float(x_point))
            self.y_pose.append(float(y_point))
            del self.x_pose[0]
            del self.y_pose[0]
            self.cur_pos.set_xdata(self.x_pose)
            self.cur_pos.set_ydata(self.y_pose)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.fig.savefig(f'{self.render_path}/{self.frame_number}.png')
    
    def plot_waypoints(self):
        x = []
        y = []
        for each_waypoint in self.waypoints_list:
            x.append(each_waypoint.x)
            y.append(each_waypoint.y)
        self.ax.plot(x, y, "+r")