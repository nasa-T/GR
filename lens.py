# import pygame as pg
from gr import *
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib import cm



class Photon:
    def __init__(self, x, vx, y, vy, z=0, vz=0, freq=1e9):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.freq = freq
        self.ut = 1 # ut of stationary observer at location of photon
        self.vt = 1
        self.vlam_x = vx*self.freq
        self.vlam_y = vy*self.freq
        self.vlam_z = vz*self.freq
        self.x_list = [x]
        self.y_list = [y]
        self.z_list = [z]
        self.e_list = []
        self.absorbed = False
        
        # assert int(np.sqrt(vx**2 + vy**2)) == 1

    def set_energy(self, energy):
        self.e_list = [energy]

    def accelerate(self, ax, ay, az, dt, vt):
        if not self.absorbed:
            dlam = dt/vt
            self.vt = vt
            self.vlam_x += ax * dlam
            self.vlam_y += ay * dlam
            self.vlam_z += az * dlam
            # print(ax*dlam, ay*dlam, az*dlam)
            # # self.freq = * vt
            self.vx = self.vlam_x / self.vt
            self.vy = self.vlam_y / self.vt
            self.vz = self.vlam_z / self.vt
            # self.vx += ax * dlam 
            # self.vy += ay * dlam 
            # self.vz += az * dlam 

    def move(self, dt):
        if not self.absorbed:
            # self.x += self.vx * dlam * self.vt
            # self.y += self.vy * dlam * self.vt
            # self.z += self.vz * dlam * self.vt
            self.x += self.vx * dt
            self.y += self.vy * dt
            self.z += self.vz * dt
            self.x_list.append(self.x)
            self.y_list.append(self.y)
            self.z_list.append(self.z)

    def absorb(self):
        self.absorbed = True

    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y) and (self.vx == other.vx) and (self.vy == other.vy) and (self.vz == other.vz) and (self.z == other.z)

class PointSource:
    def __init__(self, x, y, z=0, n_rays=8):
        self.photons = []
        for i in range(n_rays):
            vx = np.cos(i*(2*np.pi)/n_rays)
            vy = np.sin(i*(2*np.pi)/n_rays)
            self.photons.append(Photon(x, vx, y, vy))

        self.x = x
        self.y = y
        self.n_rays = n_rays
        self.lam = 0
    
    def accelerate_photons(self, gravitizer, dlam):
        for ph in self.photons:
            gravitizer.accelerate_photon(ph, dlam)
        self.lam += dlam
        
    def move_photons(self, dlam):
        for ph in self.photons:
            ph.move(dlam)

    def reset(self):
        for i in range(self.n_rays):
            vx = np.cos(i*(2*np.pi)/self.n_rays)
            vy = np.sin(i*(2*np.pi)/self.n_rays)
            ph = self.photons[i]
            ph.x, ph.y = self.x, self.y
            ph.vx, ph.vy = vx, vy
            ph.x_list = []
            ph.y_list = []
        self.lam = 0


class Flashlight:
    """
    width is the opening angle in degrees
    phi is counterclockwise angle from x-hat direction
    """
    def __init__(self, x, y, z=0, n_rays=3, width=30, phi=0, theta=90, flat=True):
        self.photons = []

        for i in range(n_rays):
            if flat:
                # angle of a light ray
                if n_rays > 1:
                    ray_angle_phi = -width/2 + i*width/(n_rays-1)
                    vx = np.cos((2*np.pi)*(ray_angle_phi+phi)/360)
                    vy = np.sin((2*np.pi)*(ray_angle_phi+phi)/360)
                    vz = 0
                else:
                    ray_angle_phi = phi
                    vx = np.cos((2*np.pi)*(ray_angle_phi)/360)
                    vy = np.sin((2*np.pi)*(ray_angle_phi)/360)
                    vz = 0
                
            else:
                if n_rays > 1:
                    ray_angle_theta = width/2
                    # -width/2 * np.cos(2*np.pi*i*width/(n_rays*2))
                    ray_angle_phi = i * 360/n_rays
                    # -width/2 * np.sin(2*np.pi*i*width/(n_rays*2))
                    
                    # vx = np.sin((2*np.pi/360)*(ray_angle_theta))*np.cos((2*np.pi/360)*(ray_angle_phi))*np.sin(2*np.pi/360*theta)*np.cos(2*np.pi/360*phi)
                    # vy = np.sin((2*np.pi/360)*(ray_angle_theta))*np.sin((2*np.pi/360)*(ray_angle_phi))*np.cos(2*np.pi/360*phi)
                    # vz = np.cos((2*np.pi/360)*(ray_angle_theta))
                    vx = np.sin((2*np.pi/360)*(ray_angle_theta))*np.cos((2*np.pi/360)*(ray_angle_phi))
                    # *np.sin(2*np.pi/360*theta)*np.cos(2*np.pi/360*phi)
                    vy = np.sin((2*np.pi/360)*(ray_angle_theta))*np.sin((2*np.pi/360)*(ray_angle_phi))
                    # *np.cos(2*np.pi/360*phi)
                    vz = np.cos((2*np.pi/360)*(ray_angle_theta))

                    # x_rot = np.cos((2*np.pi/360)*phi)*(np.cos((2*np.pi/360)*theta)*vx - np.sin((2*np.pi/360)*theta)*vz) + np.sin((2*np.pi/360)*phi)*vy
                    # y_rot = np.sin((2*np.pi/360)*phi)*(np.cos((2*np.pi/360)*theta)*vx - np.sin((2*np.pi/360)*theta)*vz)+np.cos((2*np.pi/360)*phi)*vy
                    # z_rot = np.sin((2*np.pi/360)*theta)*vx + np.cos((2*np.pi/360)*theta)*vz
                    x_rot = np.cos((2*np.pi/360)*phi)*(np.cos((2*np.pi/360)*theta)*vx + np.sin((2*np.pi/360)*theta)*vz) - np.sin((2*np.pi/360)*phi)*vy
                    y_rot = -np.sin((2*np.pi/360)*phi)*(np.cos((2*np.pi/360)*theta)*vx + np.sin((2*np.pi/360)*theta)*vz)+np.cos((2*np.pi/360)*phi)*vy
                    z_rot = -np.sin((2*np.pi/360)*theta)*vx + np.cos((2*np.pi/360)*theta)*vz
                    vx = x_rot
                    vy = y_rot
                    vz = z_rot
                else:
                    ray_angle_phi = phi
                    ray_angle_theta = theta
                    vx = np.sin((2*np.pi/360)*(ray_angle_theta))*np.cos((2*np.pi/360)*(ray_angle_phi))
                    vy = np.sin((2*np.pi/360)*(ray_angle_theta))*np.sin((2*np.pi/360)*(ray_angle_phi))
                    vz = np.cos((2*np.pi/360)*(ray_angle_theta))

                # vx = np.sin((2*np.pi)*(ray_angle_theta+theta))*np.cos((2*np.pi)*(ray_angle_phi+phi))
                # vy = np.sin((2*np.pi)*(ray_angle_theta+theta))*np.sin((2*np.pi)*(ray_angle_phi+phi))
                # vz = np.cos((2*np.pi)*(ray_angle_theta+theta))

                

            self.photons.append(Photon(x, vx, y, vy, z, vz))
        if not flat and n_rays > 1:
            # vx = np.sin(2*np.pi/360*theta)*np.sin(2*np.pi/360*phi)
            # vy = np.cos(2*np.pi/360*theta)*np.sin(2*np.pi/360*phi)
            # vx = np.cos(2*np.pi/360*phi)
            vx = np.sin(2*np.pi/360*theta)*np.cos(2*np.pi/360*phi)
            vy = np.sin(2*np.pi/360*theta)*np.sin(2*np.pi/360*phi)
            vz = np.cos(2*np.pi/360*theta)
            # self.photons.append(Photon(x, vx, y, vy, z, vz))

# np.sin((2*np.pi)*(theta)/360)
        self.x = x
        self.y = y
        self.z = z
        self.n_rays = n_rays
        self.width = width
        self.phi = phi
        self.theta = theta
        self.flat = flat
        self.lam = 0
        self.t = 0
    
    def get_photon_ut(self, photon, gravitizer):
        x = float(gravitizer.x-photon.x)
        y = float(gravitizer.y-photon.y)
        z = float(gravitizer.z-photon.z)
        r = np.sqrt(x**2 + y**2 + z**2)
        # if r <= gravitizer.rs:
        #     return 1
        if 'r' in gravitizer.metric.coord_names:
            
            th = np.arccos(z/r)
            phi = np.sign(y)*np.arccos(x/np.sqrt(x**2+y**2))
            ut = float(sp.sqrt(-1/gravitizer.metric.get_element(0,0)).subs({'r': r, 'th': th, 'phi': phi}))
        elif 'x' in gravitizer.metric.coord_names:
            ut = float(sp.sqrt(-1/gravitizer.metric.get_element(0,0)).subs({'x': x, 'y': y, 'z': z}))
        else:
            ut = 1
        return ut

    def set_photon_energy(self, gravitizer):
        # energy as measured by a stationary observer at the initial point of the photon
        # x = float(gravitizer.x-self.x)
        # y = float(gravitizer.y-self.y)
        # z = float(gravitizer.z-self.z)
        # if 'r' in gravitizer.metric.coord_names:
        #     r = np.sqrt(x**2 + y**2 + z**2)
        #     th = np.arccos(z/r)
        #     phi = np.sign(y)*np.arccos(x/np.sqrt(x**2+y**2))
        #     ut = float(sp.sqrt(-1/gravitizer.metric.get_element(0,0)).subs({'r': r, 'th': th, 'phi': phi}))
        # elif 'x' in gravitizer.metric.coord_names:
        #     ut = float(sp.sqrt(-1/gravitizer.metric.get_element(0,0)).subs({'x': x, 'y': y, 'z': z}))
        
        for ph in self.photons:
            ut = self.get_photon_ut(ph, gravitizer)
            ph.set_energy(ph.freq * ut)

    def accelerate_photons(self, gravitizer, dt):
        for ph in self.photons:
            gravitizer.accelerate_photon(ph, dt)
            if not ph.absorbed:
                ut = self.get_photon_ut(ph, gravitizer)
                
                ph.e_list.append(ph.freq * ut)
        self.t += dt
        
        
    def move_photons(self, dt):
        for ph in self.photons:
            ph.move(dt)

    def reset(self):
        self.__init__(self.x, self.y, self.z, self.n_rays, self.width, self.phi, self.theta, self.flat)
        # for i in range(self.n_rays):
        #     if self.n_rays > 1:
        #         ray_angle = self.phi-self.width/2 + i*self.width/(self.n_rays-1)
        #     else:
        #         ray_angle = self.phi
        #     vx = np.cos((2*np.pi)*ray_angle/360)
        #     vy = np.sin((2*np.pi)*ray_angle/360)
        #     ph = self.photons[i]
        #     ph.x, ph.y, ph.z = self.x, self.y, self.z
        #     ph.vx, ph.vy = vx, vy
        #     ph.x_list = [ph.x]
        #     ph.y_list = [ph.y]
        #     ph.z_list = [ph.z]
        # self.lam = 0

class FlatSpace:
    def __init__(self, x, y, z):

        tt = -1
        s = 1
        self.metric = Metric(tt, s, s, s, coords='t x y z')
        self.x = x
        self.y = y
        self.z = z

    def accelerate_photon(self, photon, dt):
        x = float(photon.x-self.x)
        y = float(photon.y-self.y)
        z = float(photon.z-self.z)
        vt = float(sp.sqrt((-self.metric.get_element(1,1)*photon.vlam_x**2 - self.metric.get_element(2,2)*photon.vlam_y**2 - self.metric.get_element(3,3)*photon.vlam_z**2)/self.metric.get_element(0,0)).subs({'x': x, 'y': y, 'z': z}))

        ax = float(-(self.metric.christoffel(1,0,0)*vt*vt + self.metric.christoffel(1,1,1)*photon.vx*photon.vx + self.metric.christoffel(1,2,2)*photon.vy*photon.vy + self.metric.christoffel(1,3,3)*photon.vz*photon.vz + 2*self.metric.christoffel(1,1,2)*photon.vx*photon.vy + 2*self.metric.christoffel(1,0,1)*vt*photon.vx + 2*self.metric.christoffel(1,0,2)*vt*photon.vy + 2*self.metric.christoffel(1,0,3)*vt*photon.vz + 2*self.metric.christoffel(1,1,3)*photon.vx*photon.vz + 2*self.metric.christoffel(1,2,3)*photon.vy*photon.vz).subs({'x': photon.x, 'y': photon.y, 'z': photon.z}))
        ay = float(-(self.metric.christoffel(2,0,0)*vt*vt + self.metric.christoffel(2,1,1)*photon.vx*photon.vx + self.metric.christoffel(2,2,2)*photon.vy*photon.vy + self.metric.christoffel(2,3,3)*photon.vz*photon.vz + 2*self.metric.christoffel(2,1,2)*photon.vx*photon.vy + 2*self.metric.christoffel(2,0,1)*vt*photon.vx + 2*self.metric.christoffel(2,0,2)*vt*photon.vy + 2*self.metric.christoffel(2,0,3)*vt*photon.vz + 2*self.metric.christoffel(2,1,3)*photon.vx*photon.vz + 2*self.metric.christoffel(2,2,3)*photon.vy*photon.vz).subs({'x': photon.x, 'y': photon.y, 'z': photon.z}))

        az = float(-(self.metric.christoffel(3,0,0)*vt*vt + self.metric.christoffel(3,1,1)*photon.vx*photon.vx + self.metric.christoffel(3,2,2)*photon.vy*photon.vy + self.metric.christoffel(3,3,3)*photon.vz*photon.vz + 2*self.metric.christoffel(3,1,2)*photon.vx*photon.vy + 2*self.metric.christoffel(3,0,1)*vt*photon.vx + 2*self.metric.christoffel(3,0,2)*vt*photon.vy + 2*self.metric.christoffel(3,0,3)*vt*photon.vz + 2*self.metric.christoffel(3,1,3)*photon.vx*photon.vz + 2*self.metric.christoffel(3,2,3)*photon.vy*photon.vz).subs({'x': photon.x, 'y': photon.y, 'z': photon.z}))

        ut = float(sp.sqrt(-1/self.metric.get_element(0,0)).subs({'x': x, 'y': y, 'z': z}))
        photon.accelerate(ax, ay, az, dt, vt)


class SchwarzschildGravitizer2D:
    def __init__(self, x, y, rs):
        r = 'sqrt((x - {0})**2 + (y - {1})**2)'.format(x, y)
        # tt = '-(1 - {0}/(4*{1}))**2/(1 + {0}/(4*{1}))**2'.format(rs, r)
        # s = '(1 + {0}/(4*{1}))**4'.format(rs, r)
        tt = '-(1 - {0}/({1}))**2/(1 + {0}/({1}))**2'.format(rs, r)
        s = '(1 + {0}/({1}))**4'.format(rs, r)
        self.metric = Metric(tt, s, s, coords='t x y')
        self.x = x
        self.y = y
        self.z = 0
        self.rs = rs

    def accelerate_photon(self, photon, dlam):
        if photon in self:
            photon.absorb()
            return
        x = float(photon.x-self.x)
        y = float(photon.y-self.y)
        vt = float(sp.sqrt((-self.metric.get_element(1,1)*photon.vlam_x**2 - self.metric.get_element(2,2)*photon.vlam_y**2)/self.metric.get_element(0,0)).subs({'x': x, 'y': y}))

        ax = float(-(self.metric.christoffel(1,0,0)*vt*vt + self.metric.christoffel(1,1,1)*photon.vx*photon.vx + self.metric.christoffel(1,2,2)*photon.vy*photon.vy + 2*self.metric.christoffel(1,1,2)*photon.vx*photon.vy + 2*self.metric.christoffel(1,0,1)*vt*photon.vx + 2*self.metric.christoffel(1,0,2)*vt*photon.vy).subs({'x': photon.x, 'y': photon.y}))
        ay = float(-(self.metric.christoffel(2,0,0)*vt*vt + self.metric.christoffel(2,1,1)*photon.vx*photon.vx + self.metric.christoffel(2,2,2)*photon.vy*photon.vy + 2 * self.metric.christoffel(2,1,2)*photon.vx*photon.vy + 2*self.metric.christoffel(2,0,1)*vt*photon.vx + 2*self.metric.christoffel(2,0,2)*vt*photon.vy).subs({'x': photon.x, 'y': photon.y}))

        ut = float(sp.sqrt(-1/self.metric.get_element(0,0)).subs({'x': x, 'y': y}))
        photon.accelerate(ax, ay, 0, dlam, vt)

    def __contains__(self, photon):
        r = np.sqrt((self.x - photon.x)**2 + (self.y - photon.y)**2)
        return r < self.rs


class SchwarzschildGravitizer3D:
    def __init__(self, x, y, z, rs):
        r = 'sqrt((x - {0})**2 + (y - {1})**2 + (z - {2})**2)'.format(x, y, z)
        tt = '-(1 - {0}/(4*{1}))**2/(1 + {0}/(4*{1}))**2'.format(rs, r)
        s = '(1 + {0}/(4*{1}))**4'.format(rs, r)
        self.metric = Metric(tt, s, s, s, coords='t x y z')
        self.x = x
        self.y = y
        self.z = z
        self.rs = rs

    def accelerate_photon(self, photon, dt):
        if photon in self:
            photon.absorb()
            return
        x = float(photon.x-self.x)
        y = float(photon.y-self.y)
        z = float(photon.z-self.z)
        vt = float(sp.sqrt((-self.metric.get_element(1,1)*photon.vlam_x**2 - self.metric.get_element(2,2)*photon.vlam_y**2 - self.metric.get_element(3,3)*photon.vlam_z**2)/self.metric.get_element(0,0)).subs({'x': x, 'y': y, 'z': z}))

        ax = float(-(self.metric.christoffel(1,0,0)*vt*vt + self.metric.christoffel(1,1,1)*photon.vx*photon.vx + self.metric.christoffel(1,2,2)*photon.vy*photon.vy + self.metric.christoffel(1,3,3)*photon.vz*photon.vz + 2*self.metric.christoffel(1,1,2)*photon.vx*photon.vy + 2*self.metric.christoffel(1,0,1)*vt*photon.vx + 2*self.metric.christoffel(1,0,2)*vt*photon.vy + 2*self.metric.christoffel(1,0,3)*vt*photon.vz + 2*self.metric.christoffel(1,1,3)*photon.vx*photon.vz + 2*self.metric.christoffel(1,2,3)*photon.vy*photon.vz).subs({'x': photon.x, 'y': photon.y, 'z': photon.z}))
        ay = float(-(self.metric.christoffel(2,0,0)*vt*vt + self.metric.christoffel(2,1,1)*photon.vx*photon.vx + self.metric.christoffel(2,2,2)*photon.vy*photon.vy + self.metric.christoffel(2,3,3)*photon.vz*photon.vz + 2*self.metric.christoffel(2,1,2)*photon.vx*photon.vy + 2*self.metric.christoffel(2,0,1)*vt*photon.vx + 2*self.metric.christoffel(2,0,2)*vt*photon.vy + 2*self.metric.christoffel(2,0,3)*vt*photon.vz + 2*self.metric.christoffel(2,1,3)*photon.vx*photon.vz + 2*self.metric.christoffel(2,2,3)*photon.vy*photon.vz).subs({'x': photon.x, 'y': photon.y, 'z': photon.z}))

        az = float(-(self.metric.christoffel(3,0,0)*vt*vt + self.metric.christoffel(3,1,1)*photon.vx*photon.vx + self.metric.christoffel(3,2,2)*photon.vy*photon.vy + self.metric.christoffel(3,3,3)*photon.vz*photon.vz + 2*self.metric.christoffel(3,1,2)*photon.vx*photon.vy + 2*self.metric.christoffel(3,0,1)*vt*photon.vx + 2*self.metric.christoffel(3,0,2)*vt*photon.vy + 2*self.metric.christoffel(3,0,3)*vt*photon.vz + 2*self.metric.christoffel(3,1,3)*photon.vx*photon.vz + 2*self.metric.christoffel(3,2,3)*photon.vy*photon.vz).subs({'x': photon.x, 'y': photon.y, 'z': photon.z}))

        ut = float(sp.sqrt(-1/self.metric.get_element(0,0)).subs({'x': x, 'y': y, 'z': z}))
        photon.accelerate(ax, ay, az, dt, vt)

    def __contains__(self, photon):
        r = np.sqrt((self.x - photon.x)**2 + (self.y - photon.y)**2 + (self.z - photon.z)**2)
        return r < self.rs

class SchwarzschildGravitizerSpherical:
    def __init__(self, x, y, z, rs):
        tt = '-(1 - {}/r)'.format(rs)
        s = '1/(1 - {}/r)'.format(rs)
        self.metric = Metric(tt, s, 'r**2', 'r**2 * sin(th)**2', coords='t r th phi')
        self.x = x
        self.y = y
        self.z = z
        self.rs = rs

    def accelerate_photon(self, photon, dt):
        if photon in self:
            photon.absorb()
            return
        x = float(photon.x-self.x)
        y = float(photon.y-self.y)
        z = float(photon.z-self.z)
        r = np.sqrt(x**2 + y**2 + z**2)
        th = np.arccos(z/r)
        phi = np.sign(y)*np.arccos(x/np.sqrt(x**2+y**2))
        vr = x/r * photon.vlam_x + y/r * photon.vlam_y + z/r * photon.vlam_z

        vth = (x*z*photon.vlam_x - x**2*photon.vlam_z + y*(z*photon.vlam_y - y*photon.vlam_z))/(np.sqrt((x**2+y**2)/r**2)*r**3)
        vphi = (-y*photon.vlam_x + x * photon.vlam_y)/(x**2 + y**2)

        vt = float(sp.sqrt((-self.metric.get_element(1,1)*vr**2 - self.metric.get_element(2,2)*vth**2 - self.metric.get_element(3,3)*vphi**2)/self.metric.get_element(0,0)).subs({'r': r, 'th': th, 'phi': phi}))

        ar = float(-(self.metric.christoffel(1,0,0)*vt*vt + self.metric.christoffel(1,1,1)*vr*vr + self.metric.christoffel(1,2,2)*vth*vth + self.metric.christoffel(1,3,3)*vphi*vphi + 2*self.metric.christoffel(1,0,1)*vt*vr + 2*self.metric.christoffel(1,0,2)*vt*vth + 2*self.metric.christoffel(1,0,3)*vt*vphi + 2*self.metric.christoffel(1,1,2)*vr*vth + 2*self.metric.christoffel(1,1,3)*vr*vphi + 2*self.metric.christoffel(1,2,3)*vth*vphi).subs({'r': r, 'th': th, 'phi': phi}))
        ath = float(-(self.metric.christoffel(2,0,0)*vt*vt + self.metric.christoffel(2,1,1)*vr*vr + self.metric.christoffel(2,2,2)*vth*vth + self.metric.christoffel(2,3,3)*vphi*vphi + 2*self.metric.christoffel(2,0,1)*vt*vr + 2*self.metric.christoffel(2,0,2)*vt*vth + 2*self.metric.christoffel(2,0,3)*vt*vphi + 2*self.metric.christoffel(2,1,2)*vr*vth + 2*self.metric.christoffel(2,1,3)*vr*vphi + 2*self.metric.christoffel(2,2,3)*vth*vphi).subs({'r': r, 'th': th, 'phi': phi}))
        aphi = float(-(self.metric.christoffel(3,0,0)*vt*vt + self.metric.christoffel(3,1,1)*vr*vr + self.metric.christoffel(3,2,2)*vth*vth + self.metric.christoffel(3,3,3)*vphi*vphi + 2*self.metric.christoffel(3,0,1)*vt*vr + 2*self.metric.christoffel(3,0,2)*vt*vth + 2*self.metric.christoffel(3,0,3)*vt*vphi + 2*self.metric.christoffel(3,1,2)*vr*vth + 2*self.metric.christoffel(3,1,3)*vr*vphi + 2*self.metric.christoffel(3,2,3)*vth*vphi).subs({'r': r, 'th': th, 'phi': phi}))

        ax = float(-np.cos(phi)*r*np.sin(th)*(vphi**2) - 2*np.sin(phi)*np.sin(th)*vphi*vr - 2*np.cos(th)*r*np.sin(phi)*vphi*vth + 2*np.cos(phi)*np.cos(th)*vr*vth - np.cos(phi)*r*np.sin(th)*(vth**2) - r*np.sin(phi)*np.sin(th)*aphi + np.cos(phi)*np.sin(th)*ar + np.cos(phi)*np.cos(th)*r*ath)
        
        ay = float(-r*np.sin(phi)*np.sin(th)*(vphi**2) + 2*np.cos(phi)*np.sin(th)*vphi*vr + 2*np.cos(phi)*np.cos(th)*r*vphi*vth + 2*np.cos(th)*np.sin(phi)*vr*vth - r*np.sin(phi)*np.sin(th)*(vth**2) + np.cos(phi)*r*np.sin(th)*aphi + np.sin(phi)*np.sin(th)*ar + np.cos(th)*r*np.sin(phi)*ath)

        az = float(-2 * np.sin(th)*vr*vth - np.cos(th)*r*(vth**2) + np.cos(th)*ar - r*np.sin(th)*ath)

        photon.accelerate(ax, ay, az, dt, vt)
    
    def __contains__(self, photon):
        r = np.sqrt((self.x - photon.x)**2 + (self.y - photon.y)**2 + (self.z - photon.z)**2)
        return r < self.rs

class KerrGravitizer:
    def __init__(self, x, y, z, rs, J):
        a = 2*J/rs
        rho2 = 'r**2 + {}**2 * cos(th)**2'.format(a)
        tr = 'r**2 - {0}*r + {1}**2'.format(rs, a)
        tt = '-(1 - {0}*r/{1})'.format(rs, rho2)
        pht = '-(2*{0}*{1}*r*sin(th)**2/{2})'.format(rs,a,rho2)
        rr = '{0}/{1}'.format(rho2,tr)
        thth = '{}'.format(rho2)
        phph = '(r**2 + {0}**2 + {1}*r*{0}**2 * sin(th)**2/{2})*sin(th)**2'.format(a,rs,rho2)
        self.metric = Metric([tt,'0','0',pht], rr, thth, [pht,'0','0',phph], coords='t r th phi')
        self.x = x
        self.y = y
        self.z = z
        self.rs = rs

    def accelerate_photon(self, photon, dt):
        if photon in self:
            photon.absorb()
            return
        x = float(photon.x-self.x)
        y = float(photon.y-self.y)
        z = float(photon.z-self.z)
        r = np.sqrt(x**2 + y**2 + z**2)
        th = np.arccos(z/r)
        phi = np.sign(y)*np.arccos(x/np.sqrt(x**2+y**2))
        vr = x/r * photon.vlam_x + y/r * photon.vlam_y + z/r * photon.vlam_z

        vth = (x*z*photon.vlam_x - x**2*photon.vlam_z + y*(z*photon.vlam_y - y*photon.vlam_z))/(np.sqrt((x**2+y**2)/r**2)*r**3)
        vphi = (-y*photon.vlam_x + x * photon.vlam_y)/(x**2 + y**2)

        # vt is now given by 0 = gtt*vt**2 + (2*gta*va)*vt + gab *va*vb
        # so we have the quadratic equation to solve for vt
        a = self.metric.get_element(0,0)
        b = 0
        c = 0
        for coord, vel in [('r',vr), ('th',vth), ('phi',vphi)]:
            b += 2 * self.metric.get_element('t',coord) * vel
            for coord2, vel2 in [('r',vr), ('th',vth), ('phi',vphi)]:
                c += self.metric.get_element(coord, coord2) * vel * vel2
        vt = float(((-b - sp.sqrt(b**2 - 4*a*c))/(2*a)).subs({'r': r, 'th': th, 'phi': phi}))
        # vt = float(sp.sqrt((-self.metric.get_element(1,1)*vr**2 - self.metric.get_element(2,2)*vth**2 - self.metric.get_element(3,3)*vphi**2)/self.metric.get_element(0,0)).subs({'r': r, 'th': th, 'phi': phi}))

        ar = float(-(self.metric.christoffel(1,0,0)*vt*vt + self.metric.christoffel(1,1,1)*vr*vr + self.metric.christoffel(1,2,2)*vth*vth + self.metric.christoffel(1,3,3)*vphi*vphi + 2*self.metric.christoffel(1,0,1)*vt*vr + 2*self.metric.christoffel(1,0,2)*vt*vth + 2*self.metric.christoffel(1,0,3)*vt*vphi + 2*self.metric.christoffel(1,1,2)*vr*vth + 2*self.metric.christoffel(1,1,3)*vr*vphi + 2*self.metric.christoffel(1,2,3)*vth*vphi).subs({'r': r, 'th': th, 'phi': phi}))
        ath = float(-(self.metric.christoffel(2,0,0)*vt*vt + self.metric.christoffel(2,1,1)*vr*vr + self.metric.christoffel(2,2,2)*vth*vth + self.metric.christoffel(2,3,3)*vphi*vphi + 2*self.metric.christoffel(2,0,1)*vt*vr + 2*self.metric.christoffel(2,0,2)*vt*vth + 2*self.metric.christoffel(2,0,3)*vt*vphi + 2*self.metric.christoffel(2,1,2)*vr*vth + 2*self.metric.christoffel(2,1,3)*vr*vphi + 2*self.metric.christoffel(2,2,3)*vth*vphi).subs({'r': r, 'th': th, 'phi': phi}))
        aphi = float(-(self.metric.christoffel(3,0,0)*vt*vt + self.metric.christoffel(3,1,1)*vr*vr + self.metric.christoffel(3,2,2)*vth*vth + self.metric.christoffel(3,3,3)*vphi*vphi + 2*self.metric.christoffel(3,0,1)*vt*vr + 2*self.metric.christoffel(3,0,2)*vt*vth + 2*self.metric.christoffel(3,0,3)*vt*vphi + 2*self.metric.christoffel(3,1,2)*vr*vth + 2*self.metric.christoffel(3,1,3)*vr*vphi + 2*self.metric.christoffel(3,2,3)*vth*vphi).subs({'r': r, 'th': th, 'phi': phi}))

        ax = float(-np.cos(phi)*r*np.sin(th)*(vphi**2) - 2*np.sin(phi)*np.sin(th)*vphi*vr - 2*np.cos(th)*r*np.sin(phi)*vphi*vth + 2*np.cos(phi)*np.cos(th)*vr*vth - np.cos(phi)*r*np.sin(th)*(vth**2) - r*np.sin(phi)*np.sin(th)*aphi + np.cos(phi)*np.sin(th)*ar + np.cos(phi)*np.cos(th)*r*ath)
        
        ay = float(-r*np.sin(phi)*np.sin(th)*(vphi**2) + 2*np.cos(phi)*np.sin(th)*vphi*vr + 2*np.cos(phi)*np.cos(th)*r*vphi*vth + 2*np.cos(th)*np.sin(phi)*vr*vth - r*np.sin(phi)*np.sin(th)*(vth**2) + np.cos(phi)*r*np.sin(th)*aphi + np.sin(phi)*np.sin(th)*ar + np.cos(th)*r*np.sin(phi)*ath)

        az = float(-2 * np.sin(th)*vr*vth - np.cos(th)*r*(vth**2) + np.cos(th)*ar - r*np.sin(th)*ath)

        photon.accelerate(ax, ay, az, dt, vt)
    
    def __contains__(self, photon):
        r = np.sqrt((self.x - photon.x)**2 + (self.y - photon.y)**2 + (self.z - photon.z)**2)
        return r < self.rs

class Simulator:
    def __init__(self, sources, gravitizer):
        self.sources = sources if type(sources) in (list, tuple) else [sources]
        self.gravitizer = gravitizer
        self.lam = 0
        self.t = 0
        for source in self.sources:
            source.set_photon_energy(gravitizer)
        # print(self.source.photons[0].e_list)

    def evolve(self, max_t, dt):
        if dt > 0:
            while self.t < max_t:
                for source in self.sources:
                    source.accelerate_photons(self.gravitizer, dt)
                    
                    source.move_photons(dt)
                
                # self.lam += dlam
                self.t += dt
        # elif dt < 0:
        #     while self.t > max_t:
        #         self.source.accelerate_photons(self.gravitizer, dt)
        #         self.source.move_photons(dt)
        #         # self.lam += dlam
        #         self.t += dt


    def draw(self, threeD=False, ax=None):
        if ax == None:
            if threeD:
                fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
                
            else:
                fig, ax = plt.subplots()
            fig.set_figwidth(8)
            fig.set_figheight(8)
        if ax.name == '3d':
            xs = []
            ys = []
            zs = []
            for source in self.sources:
                for ph in source.photons:
                    xs += ph.x_list
                    ys += ph.y_list
                    zs += ph.z_list
                    ax.scatter(ph.x_list, ph.y_list, ph.z_list, c=ph.e_list, s=8,cmap='Spectral', edgecolor='none')
                    # ax.plot(ph.x_list, ph.y_list, ph.z_list)
            if type(self.gravitizer) != FlatSpace:
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = self.gravitizer.rs*np.cos(u)*np.sin(v)
                y = self.gravitizer.rs*np.sin(u)*np.sin(v)
                z = self.gravitizer.rs*np.cos(v)
                ax.plot_surface(x, y, z, color="black")
                
                xs += list(x.flatten())
                ys += list(y.flatten())
                zs += list(z.flatten())
            ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        else:
            for source in self.sources:
                for ph in source.photons:
                    # p = np.array([ph.x_list,ph.y_list])
                    # p = p.T.reshape(-1,1,2)
                    # segments = np.concatenate([p[:-1], p[1:]], axis=1)
                    # lc = LineCollection(segments, cmap=plt.get_cmap('Spectral'), norm=plt.Normalize(np.min(ph.e_list), np.max(ph.e_list)))
                    # lc.set_array(np.array(ph.e_list))
                    # ax.add_collection(lc)
                    ax.scatter(ph.x_list, ph.y_list, c=ph.e_list, s=8,cmap='Spectral', edgecolor='none')
                    # ax.plot(ph.x_list, ph.y_list)
            if type(self.gravitizer) != FlatSpace:
                cir = mpatches.Circle((self.gravitizer.x, self.gravitizer.y), self.gravitizer.rs, color="black")
                ax.add_patch(cir)
            ax.set_aspect('equal')
            # ax.set_xlim(x.min(), x.max())
            # ax.set_ylim(-1.1, 1.1
        
        # plt.plot(grav.x, grav.y, 'o', markersize=)

    def find_focii(self):
        # basically find where photons probably crossed
        coord_lists = []
        for source in self.sources:
            for ph in source.photons:
                for ph2 in source.photons:
                    coord_lists.append([ph.x_list, ph.y_list, ph.z_list])
        
        
    def reset(self):
        self.lam = 0
        self.t = 0
        for source in self.sources:
            source.reset()
            source.set_photon_energy(self.gravitizer)
