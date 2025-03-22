import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# constants
G = 6.67430e-11
m_mars = 0.642e24 # kg
m_phobos = 1.060e16 # kg
m_deimos = 1.51e15
distance_phobos = 9400e3 # m
distance_deimos = 23460e3
speed_phobos = 2138 # m/s
speed_deimos = 1.3513e3
dt = 60 # 3600 s = 1 hr
num_steps = 5
trail_length_phobos = 400
trail_length_deimos = 1500
rho_t = 5e3 # kg/m3 terrestrial planet density (avg)
rho_g = 1e3 # kg/m3 gas planet and star (avg)
max_draw_r = 40

r_mars = int((m_mars/((4/3)*math.pi*3934))**(1/9))
#print(r_mars,3400e3)
r_phobos = int((m_phobos/((4/3)*math.pi*1861))**(1/9))
r_deimos = int((m_deimos/((4/3)*math.pi*1465))**(1/9))
#print(r_phobos,11.1e3)
r_max = r_mars

draw_r_mars = max(1,max_draw_r*(r_mars/r_max))
draw_r_phobos = max(1,max_draw_r*(r_phobos/r_max))
draw_r_deimos = max(1,max_draw_r*(r_deimos/r_max))

# initial conditions
x_phobos, y_phobos = distance_phobos, 0
vx_phobos, vy_phobos = 0, speed_phobos

x_deimos, y_deimos = distance_deimos, 0
vx_deimos, vy_deimos = 0, speed_deimos

# return x and y components of gravitational force on phobos by mars
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
ax.set_xlim(-distance_deimos * 1.2, distance_deimos * 1.2)
ax.set_ylim(-distance_deimos * 1.2, distance_deimos * 1.2)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks([]); plt.yticks([])
ax.set_title(f"stationary mars, orbiting phobos and deimos")
ax.set_aspect('equal')
ax.set_facecolor('black')

# mars
#mars = plt.Circle((0, 0), 6378e3, color='blue', label="mars")
#ax.add_patch(mars)
plt.plot(0, 0, color='r',marker='o', ms=draw_r_mars,label='mars')
#plt.plot(0, 0, color='blue',label='mars')

# phobos
phobos, = ax.plot([], [], color='g', marker='o', ms=draw_r_phobos, label="phobos")
deimos, = ax.plot([], [], color='b', marker='o', ms=draw_r_deimos, label="deimos")
trail_phobos, = ax.plot([], [], color='w', marker=',',alpha=0.25)
trail_deimos, = ax.plot([], [], color='w', marker=',',alpha=0.25)

# positions histories
positions_x_phobos = []
positions_y_phobos = []

positions_x_deimos = []
positions_y_deimos = []

def acceleration(m, x, y): # accel components
    fx, fy = get_Fg(m_mars, m, 0, 0, x, y)
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

approx_method = euler

def update(frame):
    global x_phobos, y_phobos, vx_phobos, vy_phobos
    global x_deimos, y_deimos, vx_deimos, vy_deimos

    vx_phobos, vy_phobos, x_phobos, y_phobos = approx_method(m_phobos, vx_phobos, vy_phobos, x_phobos, y_phobos)
    vx_deimos, vy_deimos, x_deimos, y_deimos = approx_method(m_deimos, vx_deimos, vy_deimos, x_deimos, y_deimos)

    # update histories
    positions_x_phobos.append(x_phobos)
    positions_y_phobos.append(y_phobos)

    #x_deimos += 1e4
    positions_x_deimos.append(x_deimos)
    positions_y_deimos.append(y_deimos)
    
    if len(positions_x_phobos) > trail_length_phobos:
        positions_x_phobos.pop(0)
        positions_y_phobos.pop(0)
    if len(positions_x_deimos) > trail_length_deimos:
        positions_x_deimos.pop(0)
        positions_y_deimos.pop(0)

    # wrap values in a list to avoid RuntimeError
    phobos.set_data([x_phobos], [y_phobos])  
    deimos.set_data([x_deimos], [y_deimos])  
    trail_phobos.set_data(positions_x_phobos, positions_y_phobos)
    trail_deimos.set_data(positions_x_deimos, positions_y_deimos)
    
    #print(math.sqrt(vx_phobos**2 + vy_phobos**2))

    #return phobos, deimos, trail_phobos, trail_deimos
    return phobos, trail_phobos, deimos, trail_deimos

# animation function
ani = animation.FuncAnimation(fig, update, frames=5, interval=1, blit=True)

# show animation
plt.legend()
plt.show()
