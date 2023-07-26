import tkinter as tk
from random import randint, uniform, random
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from random import randint, uniform, random

# MAIN INPUT
SCALE = 225  # enter 225 to see Earth's radio bubble
NUM_CIVS = 15600000

# set up display canvas
root = tk.Tk()
root.title("Milky Way galaxy")
c = tk.Canvas(root, width=1000, height=800, bg='black')
c.grid()
c.configure(scrollregion=(-500, -400, 500, 400))

# actual Milky Way dimensions (light-years)
DISC_RADIUS = 50000
DISC_HEIGHT = 1000
DISC_VOL = math.pi * DISC_RADIUS ** 2 * DISC_HEIGHT


def scale_galaxy():
    """Scale galaxy dimensions based on radio bubble size (scale)."""
    disc_radius_scaled = round(DISC_RADIUS / SCALE)
    bubble_vol = 4 / 3 * math.pi * (SCALE / 2) ** 3
    disc_vol_scaled = DISC_VOL / bubble_vol
    return disc_radius_scaled, disc_vol_scaled


def detect_prob(disc_vol_scaled):
    """Calculate probability of galactic civilizations detecting each other."""
    ratio = NUM_CIVS / disc_vol_scaled  # ratio of civs to scaled galaxy volume
    if ratio < 0.002:  # set very low ratios to probability of 0
        detection_prob = 0
    elif ratio >= 5:  # set high ratios to probability of 1
        detection_prob = 1
    else:
        detection_prob = -0.004757 * ratio ** 4 + 0.06681 * ratio ** 3 - 0.3605 * ratio ** 2 + 0.9215 * ratio + 0.00826
    return round(detection_prob, 3)


def random_polar_coordinates(disc_radius_scaled):
    """Generate uniform random (x, y) point within a disc for 2D display."""
    r = random()
    theta = uniform(0, 2 * math.pi)
    x = round(math.sqrt(r) * math.cos(theta) * disc_radius_scaled)
    y = round(math.sqrt(r) * math.sin(theta) * disc_radius_scaled)
    return x, y


def spirals(canvas, b, r, rot_fac, fuz_fac, arm):
    """Build spiral arms for tkinter display using logarithmic spiral formula.
    b = arbitrary constant in logarithmic spiral equation
    r = scaled galactic disc radius
    rot_fac = rotation factor
    fuz_fac = random shift in star position in arm, applied to 'fuzz' variable
    arm = spiral arm (0 = main arm, 1 = trailing stars)
    """
    spiral_stars = []
    fuzz = int(0.030 * abs(r))  # randomly shift star locations
    theta_max_degrees = 520
    for i in range(theta_max_degrees):
        theta = math.radians(i)
        x = r * math.exp(b * theta) * math.cos(theta + math.pi * rot_fac) + randint(-fuzz, fuzz) * fuz_fac
        y = r * math.exp(b * theta) * math.sin(theta + math.pi * rot_fac) + randint(-fuzz, fuzz) * fuz_fac
        spiral_stars.append((x, y))
    for x, y in spiral_stars:
        if arm == 0 and int(x % 2) == 0:
            canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill='white', outline='')
        elif arm == 0 and int(x % 2) != 0:
            canvas.create_oval(x - 1, y - 1, x + 1, y + 1, fill='white', outline='')
        elif arm == 1:
            canvas.create_oval(x, y, x, y, fill='white', outline='')

def star_haze(canvas, disc_radius_scaled, density):
    """Randomly distribute faint tkinter stars in galactic disc.
    disc_radius_scaled = galactic disc radius scaled to radio bubble diameter
    density = multiplier to vary number of stars posted
    """
    for i in range(0, disc_radius_scaled * density):
        x, y = random_polar_coordinates(disc_radius_scaled)
        canvas.create_text(x, y, fill='white', font=('Helvetica', '7'), text='.')

def main():
    """Calculate detection probability & post galaxy display & statistics."""
    disc_radius_scaled, disc_vol_scaled = scale_galaxy()
    detection_prob = detect_prob(disc_vol_scaled)
    
    # set up display canvas
    root = tk.Tk()
    root.title("Milky Way galaxy")
    c = tk.Canvas(root, width=1000, height=800, bg='black')
    c.grid()
    c.configure(scrollregion=(-500, -400, 500, 400))
    
    # build 4 main spiral arms & 4 trailing arms
    spirals(c, -0.3, disc_radius_scaled, 2, 1.5, 0)
    spirals(c, -0.3, disc_radius_scaled, 1.91, 1.5, 1)
    spirals(c, -0.3, -disc_radius_scaled, 2, 1.5, 0)
    spirals(c, -0.3, -disc_radius_scaled, -2.09, 1.5, 1)
    spirals(c, -0.3, -disc_radius_scaled, 0.5, 1.5, 0)
    spirals(c, -0.3, -disc_radius_scaled, 0.4, 1.5, 1)
    spirals(c, -0.3, -disc_radius_scaled, -0.5, 1.5, 0)
    spirals(c, -0.3, -disc_radius_scaled, -0.6, 1.5, 1)
    star_haze(c, disc_radius_scaled, density=8)
    # display legend
    c.create_text(-455, -360, fill='white', anchor='w', text='One Pixel = {} LY'.format(SCALE))
    c.create_text(-455, -330, fill='white', anchor='w', text='Radio Bubble Diameter = {} LY'.format(SCALE))
    c.create_text(-455, -300, fill='white', anchor='w',
                  text='Probability of detection for {:,} civilizations = {}'.format(NUM_CIVS, detection_prob))
    # post Earth's 225 LY diameter bubble and annotate
    if SCALE == 225:
        c.create_rectangle(115, 75, 116, 76, fill='red', outline='')
        c.create_text(118, 72, fill='red', anchor='w', text="<----------- Earth's Radio Bubble")

    # run tkinter loop
    root.mainloop()


def is_in_habitable_zone(x, y):
    """Check if the given (x, y) position is within the habitable zone."""
    distance_to_center = math.sqrt(x ** 2 + y ** 2)
    return 13000 <= distance_to_center <= 33000

def main():
    """Calculate detection probability & post galaxy display & statistics."""
    disc_radius_scaled, disc_vol_scaled = scale_galaxy()
    detection_prob = detect_prob(disc_vol_scaled)

    # set up display canvas
    root = tk.Tk()
    root.title("Milky Way galaxy")
    c = tk.Canvas(root, width=1000, height=800, bg='black')
    c.grid()
    c.configure(scrollregion=(-500, -400, 500, 400))

    # build 4 main spiral arms & 4 trailing arms
    spirals(c, -0.3, disc_radius_scaled, 2, 1.5, 0)
    spirals(c, -0.3, disc_radius_scaled, 1.91, 1.5, 1)
    spirals(c, -0.3, -disc_radius_scaled, 2, 1.5, 0)
    spirals(c, -0.3, -disc_radius_scaled, -2.09, 1.5, 1)
    spirals(c, -0.3, -disc_radius_scaled, 0.5, 1.5, 0)
    spirals(c, -0.3, -disc_radius_scaled, 0.4, 1.5, 1)
    spirals(c, -0.3, -disc_radius_scaled, -0.5, 1.5, 0)
    spirals(c, -0.3, -disc_radius_scaled, -0.6, 1.5, 1)

    # Create habitable zone and highlight regions
    for i in range(-500, 501, 10):
        for j in range(-400, 401, 10):
            if is_in_habitable_zone(i, j):
                c.create_rectangle(i, j, i + 10, j + 10, fill='green', outline='')

    star_haze(c, disc_radius_scaled, density=8)
    # display legend
    c.create_text(-455, -360, fill='white', anchor='w', text='One Pixel = {} LY'.format(SCALE))
    c.create_text(-455, -330, fill='white', anchor='w', text='Radio Bubble Diameter = {} LY'.format(SCALE))
    c.create_text(-455, -300, fill='white', anchor='w',
                  text='Probability of detection for {:,} civilizations = {}'.format(NUM_CIVS, detection_prob))
    # post Earth's 225 LY diameter bubble and annotate
    if SCALE == 225:
        c.create_rectangle(115, 75, 116, 76, fill='red', outline='')
        c.create_text(118, 72, fill='red', anchor='w', text="<----------- Earth's Radio Bubble")

    # run tkinter loop
    root.mainloop()


if __name__ == '__main__':
    main()


# Define the constants for probability calculation
NUM_EQUIV_VOLUMES = 1000  # number of locations in which to place civilizations
MAX_CIVS = 5000  # maximum number of advanced civilizations
TRIALS = 1000  # number of times to model a given number of civilizations
CIV_STEP_SIZE = 100  # civilizations count step size

# Drake equation factors (arbitrary values)
R = 10
fp = 0.5
ne = 2
fl = 0.1
fi = 0.01
fc = 0.001
L = 10000

# Number of civilizations based on the Drake equation
num_transmitting_civilizations = R * fp * ne * fl * fi * fc * L

# Convert num_transmitting_civilizations to an integer
num_transmitting_civilizations = int(num_transmitting_civilizations)

# Milky Way size (in light-years)
milky_way_size = 100000

# Radio bubble size range (in light-years)
min_radio_bubble_radius = 50
max_radio_bubble_radius = 500

# Function to generate random positions for civilizations
def generate_civilization_positions(num_civilizations, galaxy_size):
    positions = []
    for _ in range(num_civilizations):
        x = random.uniform(-galaxy_size / 2, galaxy_size / 2)
        y = random.uniform(-galaxy_size / 2, galaxy_size / 2)
        positions.append((x, y))
    return positions

# Function to estimate the probability of detection between civilizations
def estimate_detection_probability(num_civilizations, min_radius, max_radius, galaxy_size):
    total_volume_covered = 0
    for _ in range(num_civilizations):
        radius = random.uniform(min_radius, max_radius)
        total_volume_covered += (4 / 3) * math.pi * radius ** 3

    galaxy_volume = (4 / 3) * math.pi * (galaxy_size / 2) ** 3
    detection_probability = total_volume_covered / galaxy_volume
    return detection_probability

# Function to calculate the probability of 2+ civilizations per location
def calculate_probability(num_civs):
    civs_per_vol = num_civs / NUM_EQUIV_VOLUMES
    num_single_civs = 0
    for _ in range(TRIALS):
        locations = []  # equivalent volumes containing a civilization
        while len(locations) < num_civs:
            location = randint(1, NUM_EQUIV_VOLUMES)  # Use randint from random module
            locations.append(location)
        overlap_count = Counter(locations)
        overlap_rollup = Counter(overlap_count.values())
        num_single_civs += overlap_rollup[1]
    prob = 1 - (num_single_civs / (num_civs * TRIALS))
    return civs_per_vol, prob

# Generate civilization positions
civilization_positions = generate_civilization_positions(num_transmitting_civilizations, milky_way_size)

# Loop through different numbers of civilizations and calculate the probabilities
x = []  # x values for polynomial fit
y = []  # y values for polynomial fit

for num_civs in range(2, MAX_CIVS + 2, CIV_STEP_SIZE):
    civs_per_vol, prob = calculate_probability(num_civs)
    print("{:.4f} {:.4f}".format(civs_per_vol, prob))
    x.append(civs_per_vol)
    y.append(prob)

# Perform a 4th order polynomial fit and plot the results
coefficients = np.polyfit(x, y, 4)
p = np.poly1d(coefficients)
print("\n{}".format(p))

xp = np.linspace(0, 5)
plt.plot(x, y, '.', xp, p(xp), '-')
plt.ylim(-0.5, 1.5)
plt.xlabel('Civilizations per Equivalent Volume')
plt.ylabel('Probability of 2+ Civilizations per Location')
plt.title('Probability of Multiple Civilizations per Location')
plt.grid(True)
plt.show()

# Start the galaxy display and detection probability
main()
