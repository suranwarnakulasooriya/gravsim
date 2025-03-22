import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
m_earth = 5.972e24 # kg
m_moon = 7.342e22 # kg
distance = 384400000 # m
speed = 1023 # m/s
dt = 3600 # 3600 s = 1 hr
num_steps = 5
trail_length = 1.9*math.pi*distance/(speed*dt)
rho_t = 5e3 # kg/m3 terrestrial planet density (avg)
rho_g = 1e3 # kg/m3 gas planet and star (avg)
max_draw_r = 20

r_earth = 6378e3
r_moon = 1740e3
r_max = r_earth


draw_r_earth = max_draw_r*(r_earth/r_max)
draw_r_moon = max_draw_r*(r_moon/r_max)

# initial conditions
x_moon, y_moon = distance, 0
vx_moon, vy_moon = 0, speed

# return x and y components of gravitational force on moon by earth
def get_Fg(m1, m2, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    r = math.sqrt(dx**2 + dy**2)
    
    force = G * m1 * m2 / r**2
    fx = -force * dx / r
    fy = -force * dy / r
    return fx, fy

# figure setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-distance * 1.2, distance * 1.2)
ax.set_ylim(-distance * 1.2, distance * 1.2)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks([]); plt.yticks([])
ax.set_title(f"stationary earth, orbiting moon")
ax.set_aspect('equal')
ax.set_facecolor(blk)

# earth
#earth = plt.Circle((0, 0), 6378e3, color='blue', label="Earth")
#ax.add_patch(earth)
plt.plot(0, 0, color=blu,marker='o', ms=draw_r_earth)
#plt.plot(0, 0, color='blue',label='earth')

# moon
moon, = ax.plot([], [], color=wht, marker='o', ms=draw_r_moon, label="Moon")
trail, = ax.plot([], [], color=wht, marker=',',alpha=0.25)

# positions histories
positions_x = []
positions_y = []

def acceleration(m, x, y): # accel components
    fx, fy = get_Fg(m_earth, m, 0, 0, x, y)
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
    global x_moon, y_moon, vx_moon, vy_moon

    vx_moon, vy_moon, x_moon, y_moon = approx_method(m_moon, vx_moon, vy_moon, x_moon, y_moon)

    # update histories
    positions_x.append(x_moon)
    positions_y.append(y_moon)
    
    if len(positions_x) > trail_length:
        positions_x.pop(0)
        positions_y.pop(0)

    # wrap values in a list to avoid RuntimeError
    moon.set_data([x_moon], [y_moon])  
    trail.set_data(positions_x, positions_y)
    
    return moon, trail

# animation function
ani = animation.FuncAnimation(fig, update, frames=5, interval=1, blit=True)

# show animation
plt.legend()
plt.show()
