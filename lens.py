import pygame as pg
from gr import *
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Photon:
    def __init__(self, x, vx, y, vy, z=0, vz=0):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.x_list = [x]
        self.y_list = [y]
        self.z_list = [z]
        self.absorbed = False
        # assert int(np.sqrt(vx**2 + vy**2)) == 1

    def accelerate(self, ax, ay, az, dlam):
        if not self.absorbed:
            self.vx += ax * dlam
            self.vy += ay * dlam
            self.vz += az * dlam

    def move(self, dlam):
        if not self.absorbed:
            self.x += self.vx * dlam
            self.y += self.vy * dlam
            self.z += self.vz * dlam
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
    direction is counterclockwise angle from x-hat direction
    """
    def __init__(self, x, y, z=0, n_rays=3, width=30, phi=0, theta=90, flat=True):
        self.photons = []

        for i in range(n_rays):
            if flat:
                # angle of a light ray
                if n_rays > 1:
                    ray_angle_phi = -width/2 + i*width/(n_rays-1)
                    
                else:
                    ray_angle_phi = phi

                vx = np.cos((2*np.pi)*(ray_angle_phi+phi)/360)
                vy = np.sin((2*np.pi)*(ray_angle_phi+phi)/360)
                vz = 0
            else:
                if n_rays > 1:
                    ray_angle_phi = -width/2 * np.cos(2*np.pi*i*width/(n_rays*2))
                    ray_angle_theta = -width/2 * np.sin(2*np.pi*i*width/(n_rays*2))
                else:
                    ray_angle_phi = phi
                    ray_angle_theta = theta

                vx = np.sin((2*np.pi)*(ray_angle_theta+theta))*np.cos((2*np.pi)*(ray_angle_phi+phi))
                vy = np.sin((2*np.pi)*(ray_angle_theta+theta))*np.sin((2*np.pi)*(ray_angle_phi+phi))
                vz = np.cos((2*np.pi)*(ray_angle_theta+theta))
            self.photons.append(Photon(x, vx, y, vy, z, vz))
        if not flat and n_rays > 1:
            vx = np.sin(theta)*np.cos(phi)
            vy = np.sin(theta)*np.sin(phi)
            vz = np.cos(theta)
            self.photons.append(Photon(x, vx, y, vy, z, vz))

# np.sin((2*np.pi)*(theta)/360)
        self.x = x
        self.y = y
        self.z = z
        self.n_rays = n_rays
        self.width = width
        self.phi = phi
        self.theta = theta
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
            if self.n_rays > 1:
                ray_angle = self.phi-self.width/2 + i*self.width/(self.n_rays-1)
            else:
                ray_angle = self.phi
            vx = np.cos((2*np.pi)*ray_angle/360)
            vy = np.sin((2*np.pi)*ray_angle/360)
            ph = self.photons[i]
            ph.x, ph.y, ph.z = self.x, self.y, self.z
            ph.vx, ph.vy = vx, vy
            ph.x_list = [ph.x]
            ph.y_list = [ph.y]
            ph.z_list = [ph.z]
        self.lam = 0


class SchwarzschildGravitizer2D:
    def __init__(self, x, y, rs):
        r = 'sqrt((x - {0})**2 + (y - {1})**2)'.format(x, y)
        tt = '-(1 - {0}/(4*{1}))**2/(1 + {0}/(4*{1}))**2'.format(rs, r)
        s = '(1 + {0}/(4*{1}))**4'.format(rs, r)
        self.metric = Metric(tt, s, s, coords='t x y')
        self.x = x
        self.y = y
        self.rs = rs

    def accelerate_photon(self, photon, dlam):
        vt = sp.sqrt((-self.metric.get_element(1,1)*photon.vx**2 - self.metric.get_element(2,2)*photon.vy**2)/self.metric.get_element(0,0)).subs({'x': photon.x, 'y': photon.y})
        # ax = float(-(self.metric.christoffel(1,0,0, x=photon.x, y=photon.y)*vt*vt + self.metric.christoffel(1,1,1, x=photon.x, y=photon.y)*photon.vx*photon.vx + self.metric.christoffel(1,2,2, x=photon.x, y=photon.y)*photon.vy*photon.vy + 2 * self.metric.christoffel(1,1,2, x=photon.x, y=photon.y)*photon.vx*photon.vy + 2*self.metric.christoffel(1,0,1, x=photon.x, y=photon.y)*vt*photon.vx + 2*self.metric.christoffel(1,0,2, x=photon.x, y=photon.y)*vt*photon.vy))
        # ay = float(-(self.metric.christoffel(2,0,0, x=photon.x, y=photon.y)*vt*vt + self.metric.christoffel(2,1,1, x=photon.x, y=photon.y)*photon.vx*photon.vx + self.metric.christoffel(2,2,2, x=photon.x, y=photon.y)*photon.vy*photon.vy + 2 * self.metric.christoffel(2,1,2, x=photon.x, y=photon.y)*photon.vx*photon.vy + 2*self.metric.christoffel(2,0,1, x=photon.x, y=photon.y)*vt*photon.vx + 2*self.metric.christoffel(2,0,2, x=photon.x, y=photon.y)*vt*photon.vy))

        ax = float(-(self.metric.christoffel(1,0,0)*vt*vt + self.metric.christoffel(1,1,1)*photon.vx*photon.vx + self.metric.christoffel(1,2,2)*photon.vy*photon.vy + 2*self.metric.christoffel(1,1,2)*photon.vx*photon.vy + 2*self.metric.christoffel(1,0,1)*vt*photon.vx + 2*self.metric.christoffel(1,0,2)*vt*photon.vy).subs({'x': photon.x, 'y': photon.y}))
        ay = float(-(self.metric.christoffel(2,0,0)*vt*vt + self.metric.christoffel(2,1,1)*photon.vx*photon.vx + self.metric.christoffel(2,2,2)*photon.vy*photon.vy + 2 * self.metric.christoffel(2,1,2)*photon.vx*photon.vy + 2*self.metric.christoffel(2,0,1)*vt*photon.vx + 2*self.metric.christoffel(2,0,2)*vt*photon.vy).subs({'x': photon.x, 'y': photon.y}))
        # print(vt,ax,ay)
        # ax = float(-(self.metric.christoffel(1,1,1)*photon.vx*photon.vx + self.metric.christoffel(1,2,2)*photon.vy*photon.vy + 2*self.metric.christoffel(1,1,2)*photon.vx*photon.vy).subs({'x': photon.x, 'y': photon.y}))
        # ay = float(-(self.metric.christoffel(2,1,1)*photon.vx*photon.vx + self.metric.christoffel(2,2,2)*photon.vy*photon.vy + 2*self.metric.christoffel(2,1,2)*photon.vx*photon.vy).subs({'x': photon.x, 'y': photon.y}))

        photon.accelerate(ax, ay, 0, dlam)

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

    def accelerate_photon(self, photon, dlam):
        vt = sp.sqrt((-self.metric.get_element(1,1)*photon.vx**2 - self.metric.get_element(2,2)*photon.vy**2 - self.metric.get_element(3,3)*photon.vz**2)/self.metric.get_element(0,0)).subs({'x': photon.x, 'y': photon.y, 'z': photon.z})
        # ax = float(-(self.metric.christoffel(1,0,0, x=photon.x, y=photon.y)*vt*vt + self.metric.christoffel(1,1,1, x=photon.x, y=photon.y)*photon.vx*photon.vx + self.metric.christoffel(1,2,2, x=photon.x, y=photon.y)*photon.vy*photon.vy + 2 * self.metric.christoffel(1,1,2, x=photon.x, y=photon.y)*photon.vx*photon.vy + 2*self.metric.christoffel(1,0,1, x=photon.x, y=photon.y)*vt*photon.vx + 2*self.metric.christoffel(1,0,2, x=photon.x, y=photon.y)*vt*photon.vy))
        # ay = float(-(self.metric.christoffel(2,0,0, x=photon.x, y=photon.y)*vt*vt + self.metric.christoffel(2,1,1, x=photon.x, y=photon.y)*photon.vx*photon.vx + self.metric.christoffel(2,2,2, x=photon.x, y=photon.y)*photon.vy*photon.vy + 2 * self.metric.christoffel(2,1,2, x=photon.x, y=photon.y)*photon.vx*photon.vy + 2*self.metric.christoffel(2,0,1, x=photon.x, y=photon.y)*vt*photon.vx + 2*self.metric.christoffel(2,0,2, x=photon.x, y=photon.y)*vt*photon.vy))

        ax = float(-(self.metric.christoffel(1,0,0)*vt*vt + self.metric.christoffel(1,1,1)*photon.vx*photon.vx + self.metric.christoffel(1,2,2)*photon.vy*photon.vy + self.metric.christoffel(1,3,3)*photon.vz*photon.vz + 2*self.metric.christoffel(1,1,2)*photon.vx*photon.vy + 2*self.metric.christoffel(1,0,1)*vt*photon.vx + 2*self.metric.christoffel(1,0,2)*vt*photon.vy + 2*self.metric.christoffel(1,0,3)*vt*photon.vz + 2*self.metric.christoffel(1,1,3)*photon.vx*photon.vz + 2*self.metric.christoffel(1,2,3)*photon.vy*photon.vz).subs({'x': photon.x, 'y': photon.y, 'z': photon.z}))
        ay = float(-(self.metric.christoffel(2,0,0)*vt*vt + self.metric.christoffel(2,1,1)*photon.vx*photon.vx + self.metric.christoffel(2,2,2)*photon.vy*photon.vy + self.metric.christoffel(2,3,3)*photon.vz*photon.vz + 2*self.metric.christoffel(2,1,2)*photon.vx*photon.vy + 2*self.metric.christoffel(2,0,1)*vt*photon.vx + 2*self.metric.christoffel(2,0,2)*vt*photon.vy + 2*self.metric.christoffel(2,0,3)*vt*photon.vz + 2*self.metric.christoffel(2,1,3)*photon.vx*photon.vz + 2*self.metric.christoffel(2,2,3)*photon.vy*photon.vz).subs({'x': photon.x, 'y': photon.y, 'z': photon.z}))

        az = float(-(self.metric.christoffel(3,0,0)*vt*vt + self.metric.christoffel(3,1,1)*photon.vx*photon.vx + self.metric.christoffel(3,2,2)*photon.vy*photon.vy + self.metric.christoffel(3,3,3)*photon.vz*photon.vz + 2*self.metric.christoffel(3,1,2)*photon.vx*photon.vy + 2*self.metric.christoffel(3,0,1)*vt*photon.vx + 2*self.metric.christoffel(3,0,2)*vt*photon.vy + 2*self.metric.christoffel(3,0,3)*vt*photon.vz + 2*self.metric.christoffel(3,1,3)*photon.vx*photon.vz + 2*self.metric.christoffel(3,2,3)*photon.vy*photon.vz).subs({'x': photon.x, 'y': photon.y, 'z': photon.z}))

        # ax = float(-(self.metric.christoffel(1,1,1)*photon.vx*photon.vx + self.metric.christoffel(1,2,2)*photon.vy*photon.vy + 2*self.metric.christoffel(1,1,2)*photon.vx*photon.vy).subs({'x': photon.x, 'y': photon.y}))
        # ay = float(-(self.metric.christoffel(2,1,1)*photon.vx*photon.vx + self.metric.christoffel(2,2,2)*photon.vy*photon.vy + 2*self.metric.christoffel(2,1,2)*photon.vx*photon.vy).subs({'x': photon.x, 'y': photon.y}))

        photon.accelerate(ax, ay, 0, dlam)

class SchwarzschildGravitizerSpherical:
    def __init__(self, x, y, z, rs):
        # r = 'sqrt((x - {0})**2 + (y - {1})**2 + (z - {2})**2)'.format(x, y, z)
        tt = '-(1 - {0}/r)'.format(rs)
        s = '1/(1 - {0}/r)'.format(rs)
        self.metric = Metric(tt, s, 'r**2', 'r**2 * sin(th)**2', coords='t r th phi')
        self.x = x
        self.y = y
        self.z = z
        self.rs = rs

    def accelerate_photon(self, photon, dlam):
        x = float(photon.x-self.x)
        y = float(photon.y-self.y)
        z = float(photon.z-self.z)
        r = np.sqrt(x**2 + y**2 + z**2)
        th = np.arccos(z/r)
        phi = np.sign(y)*np.arccos(x/np.sqrt(x**2+y**2))
        # phi = np.arctan(y/x)
        vr = x/r * photon.vx + y/r * photon.vy + z/r * photon.vz
        vth = x*z/sp.sqrt(1-(z/r)**2) *1/r**3 * photon.vx + y*z/sp.sqrt(1-(z/r)**2) *1/r**3 * photon.vy + ((z**2/r**3) - 1/r)/sp.sqrt(1-(z/r)**2) * photon.vz
        vphi = -y/(x**2 * (1+(y/x)**2)) * photon.vx + 1/(x * (1+(y/x)**2)) * photon.vy
        vt = sp.sqrt((-self.metric.get_element(1,1)*vr**2 - self.metric.get_element(2,2)*vth**2 - self.metric.get_element(3,3)*vphi**2)/self.metric.get_element(0,0)).subs({'r': r, 'th': th, 'phi': phi})
        # ax = float(-(self.metric.christoffel(1,0,0, x=photon.x, y=photon.y)*vt*vt + self.metric.christoffel(1,1,1, x=photon.x, y=photon.y)*photon.vx*photon.vx + self.metric.christoffel(1,2,2, x=photon.x, y=photon.y)*photon.vy*photon.vy + 2 * self.metric.christoffel(1,1,2, x=photon.x, y=photon.y)*photon.vx*photon.vy + 2*self.metric.christoffel(1,0,1, x=photon.x, y=photon.y)*vt*photon.vx + 2*self.metric.christoffel(1,0,2, x=photon.x, y=photon.y)*vt*photon.vy))
        # ay = float(-(self.metric.christoffel(2,0,0, x=photon.x, y=photon.y)*vt*vt + self.metric.christoffel(2,1,1, x=photon.x, y=photon.y)*photon.vx*photon.vx + self.metric.christoffel(2,2,2, x=photon.x, y=photon.y)*photon.vy*photon.vy + 2 * self.metric.christoffel(2,1,2, x=photon.x, y=photon.y)*photon.vx*photon.vy + 2*self.metric.christoffel(2,0,1, x=photon.x, y=photon.y)*vt*photon.vx + 2*self.metric.christoffel(2,0,2, x=photon.x, y=photon.y)*vt*photon.vy))

        ar = float(-(self.metric.christoffel(1,0,0)*vt*vt + self.metric.christoffel(1,1,1)*vr*vr + self.metric.christoffel(1,2,2)*vth*vth + self.metric.christoffel(1,3,3)*vphi*vphi + 2*self.metric.christoffel(1,0,1)*vt*vr + 2*self.metric.christoffel(1,0,2)*vt*vth + 2*self.metric.christoffel(1,0,3)*vt*vphi + 2*self.metric.christoffel(1,1,2)*vr*vth + 2*self.metric.christoffel(1,1,3)*vr*vphi + 2*self.metric.christoffel(1,2,3)*vth*vphi).subs({'r': r, 'th': th, 'phi': phi}))
        ath = float(-(self.metric.christoffel(2,0,0)*vt*vt + self.metric.christoffel(2,1,1)*vr*vr + self.metric.christoffel(2,2,2)*vth*vth + self.metric.christoffel(2,3,3)*vphi*vphi + 2*self.metric.christoffel(2,0,1)*vt*vr + 2*self.metric.christoffel(2,0,2)*vt*vth + 2*self.metric.christoffel(2,0,3)*vt*vphi + 2*self.metric.christoffel(2,1,2)*vr*vth + 2*self.metric.christoffel(2,1,3)*vr*vphi + 2*self.metric.christoffel(2,2,3)*vth*vphi).subs({'r': r, 'th': th, 'phi': phi}))
        aphi = float(-(self.metric.christoffel(3,0,0)*vt*vt + self.metric.christoffel(3,1,1)*vr*vr + self.metric.christoffel(3,2,2)*vth*vth + self.metric.christoffel(3,3,3)*vphi*vphi + 2*self.metric.christoffel(3,0,1)*vt*vr + 2*self.metric.christoffel(3,0,2)*vt*vth + 2*self.metric.christoffel(3,0,3)*vt*vphi + 2*self.metric.christoffel(3,1,2)*vr*vth + 2*self.metric.christoffel(3,1,3)*vr*vphi + 2*self.metric.christoffel(3,2,3)*vth*vphi).subs({'r': r, 'th': th, 'phi': phi}))

        ax = float(-np.cos(phi)*r*np.sin(th)*(vphi**2) - 2*np.sin(phi)*np.sin(th)*vphi*vr - 2*np.cos(th)*r*np.sin(phi)*vphi*vth + 2*np.cos(phi)*np.cos(th)*vr*vth - np.cos(phi)*r*np.sin(th)*(vth**2) - r*np.sin(phi)*np.sin(th)*aphi + np.cos(phi)*np.sin(th)*ar + np.cos(phi)*np.cos(th)*r*ath)

        ay = float(-r*np.sin(phi)*np.sin(th)*(vphi**2) + 2*np.cos(phi)*np.sin(th)*vphi*vr + 2*np.cos(phi)*np.cos(th)*r*vphi*vth + 2*np.cos(th)*np.sin(phi)*vr*vth - r*np.sin(phi)*np.sin(th)*(vth**2) + np.cos(phi)*r*np.sin(th)*aphi + np.sin(phi)*np.sin(th)*ar + np.cos(th)*r*np.sin(phi)*ath)

        az = float(-2 * np.sin(th)*vr*vth - np.cos(th)*r*(vth**2) + np.cos(th)*ar - r*np.sin(th)*ath)

        photon.accelerate(ax, ay, az, dlam)

class Simulator:
    def __init__(self, source, gravitizer):
        self.source = source
        self.gravitizer = gravitizer
        self.lam = 0

    def evolve(self, maxlam, dlam):
        while self.lam < maxlam:
            self.source.accelerate_photons(self.gravitizer, dlam)
            self.source.move_photons(dlam)
            self.lam += dlam

    def draw(self, threeD=False, ax=None):
        if ax == None:
            if threeD:
                fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
                
            else:
                fig, ax = plt.subplots()
        fig.set_figwidth(8)
        fig.set_figheight(8)
        if ax.name == '3d':
            for ph in self.source.photons:
                ax.plot(ph.x_list, ph.y_list, ph.z_list)
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = self.gravitizer.rs*np.cos(u)*np.sin(v)
            y = self.gravitizer.rs*np.sin(u)*np.sin(v)
            z = self.gravitizer.rs*np.cos(v)
            ax.plot_wireframe(x, y, z, color="black")
        else:
            for ph in self.source.photons:
                ax.plot(ph.x_list, ph.y_list)
            cir = mpatches.Circle((self.gravitizer.x, self.gravitizer.y), self.gravitizer.rs/4, fill=None)
            ax.add_patch(cir)
        
        # plt.plot(grav.x, grav.y, 'o', markersize=)
        
    def reset(self):
        self.lam = 0
        self.source.reset()
