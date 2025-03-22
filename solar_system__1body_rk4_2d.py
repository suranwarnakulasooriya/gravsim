from math import pi, sqrt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime

# colors
blk = '#282828'
red = '#cc6666'
grn = '#b5bd68'
ylw = '#f0c674'
blu = '#81a2be'
mgt = '#b294bb'
cyn = '#8abeb7'
wht = '#c5c8c6'
crm = '#ebcea2'
orn = '#e6894f'

# constants
G = 6.67430e-11
m_sun = 1.9891e30 # kg
dt = 3600*24*7 # 3600 s = 1 hr
num_steps = 5

masses = [0.330e24,4.87e24,5.97e24,0.642e24,1898e24,568e24,86.8e24,102e24] #
n = 6
positions = [[57.9e9,0],[108.2e9,0],[149.6e9,0],[228e9,0],[778.5e9,0],[1432e9,0],[2867e9,0],[4515e9,0]] #
vels = [[0,47.4e3],[0,35e3],[0,29.8e3],[0,24.1e3],[0,13.1e3],[0,9.7e3],[0,6.8e3],[0,5.4e3]] #
accels = [[0,0]]*n
distances = [sqrt(x**2+y**2) for x,y in positions]
d_max = max(distances[:n])
speeds = [sqrt(vx**2+vy**2) for vx,vy in vels]
trail_lengths = [1.9*pi*distances[i]/(speeds[i]*dt) for i in range(n)]
densities = [5429,5243,5514,3934,1326,687,1270,1638] #
colors = [wht,ylw,blu,red,orn,mgt,cyn,blu]
names = ['mercury','venus','earth','mars','jupiter','saturn','uranus','neptune']
histories = [[] for i in range(n)] #
max_draw_r = 10 #
scale_exaggeration = 10 # default is 3

m_max = m_sun
radii = [(masses[i]/((4/3)*pi*densities[i]))**(1/scale_exaggeration) for i in range(n)]
r_max = max(radii)
r_sun = (m_sun/((4/3)*pi*3934))**(1/scale_exaggeration)
r_max = r_sun
draw_r_sun = max(1,max_draw_r*(r_sun/r_max))
draw_radii = [max(1,max_draw_r*(radii[i]/r_max)) for i in range(n)]

draw_trails = True

# return x and y components of gravitational force on phobos by mars
def get_Fg(m1, m2, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    r = sqrt(dx**2 + dy**2)
    
    force = G * m1 * m2 / r**2
    fx = -force * dx / r
    fy = -force * dy / r
    return fx, fy

# figure setup
scale = 2
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-d_max * scale, d_max * scale)
ax.set_ylim(-d_max * scale, d_max * scale)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks([]); plt.yticks([])
ax.set_title(f"the solar system but only the sun acts on the planets (rk4)")
ax.set_aspect('equal')
ax.set_facecolor(blk)
#title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
#                transform=ax.transAxes, ha="center")


plt.plot(0, 0, color=orn,marker='o', ms=draw_r_sun) # SUN

bodies = []
trails = []

for i in range(n):
    bodies.append(ax.plot([], [], color=colors[i], marker='o', ms=draw_radii[i], label=names[i])[0])
    if draw_trails:
        trails.append(ax.plot([], [], color=crm, marker=',',alpha=0.25)[0])


def acceleration(m, x, y): # accel components
    fx, fy = get_Fg(m_sun, m, 0, 0, x, y)
    return fx / m, fy / m

def euler(m,vx,vy,x,y): # euler's method
    ax, ay = acceleration(m,x,y) # accel components
    new_vx = vx + ax * dt; new_vy = vy + ay * dt # new velocities
    new_x = x + new_vx * dt; new_y = y + new_vy * dt # new positions
    return new_vx, new_vy, new_x, new_y

def rk4(m,vx,vy,x,y): # 4th order runge-kutta method
    ax1, ay1 = acceleration(m, x, y)

    k1vx, k1vy = ax1 * dt, ay1 * dt
    k1x, k1y = vx * dt, vy * dt

    ax2, ay2 = acceleration(m, x + k1x / 2, y + k1y / 2)
    k2vx, k2vy = ax2 * dt, ay2 * dt
    k2x, k2y = (vx + k1vx / 2) * dt, (vy + k1vy / 2) * dt

    ax3, ay3 = acceleration(m, x + k2x / 2, y + k2y / 2)
    k3vx, k3vy = ax3 * dt, ay3 * dt
    k3x, k3y = (vx + k2vx / 2) * dt, (vy + k2vy / 2) * dt

    ax4, ay4 = acceleration(m, x + k3x, y + k3y)
    k4vx, k4vy = ax4 * dt, ay4 * dt
    k4x, k4y = (vx + k3vx) * dt, (vy + k3vy) * dt

    new_vx = vx + (k1vx + 2 * k2vx + 2 * k3vx + k4vx) / 6
    new_vy = vy + (k1vy + 2 * k2vy + 2 * k3vy + k4vy) / 6
    new_x = x + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    new_y = y + (k1y + 2 * k2y + 2 * k3y + k4y) / 6

    return new_vx, new_vy, new_x, new_y

approx_method = rk4

def update(frame):
    
    #dt1 = datetime.now()
    for i in range(n):
        vels[i][0], vels[i][1], positions[i][0], positions[i][1] = approx_method(masses[i], vels[i][0], vels[i][1], positions[i][0], positions[i][1])
        histories[i].append([positions[i][0],positions[i][1]])
        if len(histories[i]) > trail_lengths[i]:
            histories[i].pop(0)
        bodies[i].set_data([positions[i][0]], [positions[i][1]])
        if draw_trails:
            trails[i].set_data([p[0] for p in histories[i]], [p[1] for p in histories[i]])
    #print((datetime.now()-dt1).microseconds)
    #title.set_text(str(datetime.now()))

    #return bodies[0],bodies[1],bodies[2],bodies[3],trails[0],trails[1],trails[2],trails[3]
    return bodies+trails if draw_trails else bodies

# animation function
ani = animation.FuncAnimation(fig, update, frames=5, interval=1, blit=True)

# show animation
#plt.legend()
#print(chr(27) + "[2J")
plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
try: plt.show()
except KeyboardInterrupt: print(' exited.');exit()
