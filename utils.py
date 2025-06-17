def sample_gaussian_data(n_samples, cov_eigvals, target_coeffs, noise_std=0):

    Phi = np.random.randn(n_samples, len(cov_eigvals))
    X = Phi * cov_eigvals ** .5
    Y = Phi @ target_coeffs

    if noise_std > 0:
      Y += noise_std * np.random.randn(*Y.shape)

    if len(Y.shape) == 1:
      Y = Y.reshape(-1, 1)

    return X, Y

def logspaced_integers(a, b, num):
    assert b - a >= num

    num_extra_pts = 0
    vals = np.unique(np.round(np.logspace(np.log10(a), np.log10(b), num + num_extra_pts)).astype(int))
    while len(vals) < num:
      num_extra_pts += 1
      vals = np.unique(np.round(np.logspace(np.log10(a), np.log10(b), num + num_extra_pts)).astype(int))
    return vals

def make_animation(plot_fn, indices, filename="animation.gif", frame_duration=0.5, **kwargs):
    import os
    import matplotlib
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from imageio.plugins.ffmpeg import FfmpegFormat
    from tqdm import tqdm

    plt.ioff()  # Turn off interactive mode to prevent automatic rendering

    # Create 'frames' directory if it doesn't exist
    if not os.path.exists('frames'):
        os.makedirs('frames')
    matplotlib.rcParams['figure.max_open_warning'] = 1000

    # List to hold frame filenames
    frame_files = []

    # For each index
    for idx, i in enumerate(tqdm(indices)):
        fig = plt.figure()  # Create a new figure
        # Run the plotting function
        plot_fn(i, **kwargs)

        # Save the plot as 'frames/frame_{idx:04d}.png'
        frame_filename = f'frames/frame_{idx:04d}.png'
        plt.savefig(frame_filename)
        plt.close(fig)  # Close the specific figure to free memory

        frame_files.append(frame_filename)

    # Read the saved images
    images = []
    for frame_filename in frame_files:
        images.append(imageio.imread(frame_filename))

    # Save the animation based on the file extension
    if filename.endswith('.gif'):
        # Save as a GIF
        imageio.mimsave(filename, images, duration=frame_duration, loop=0)
    elif filename.endswith('.mov'):
        # Save as a MOV file
        writer = imageio.get_writer(filename, format='FFMPEG', mode='I', fps=int(1 / frame_duration))
        for image in images:
            writer.append_data(image)
        writer.close()

    plt.clf()
    plt.close('all')

    # Turn interactive mode back on (optional, if you want further plots to display interactively)
    plt.ion()


# PLOTTING STUFF

def rcsetup():
    plt.rc("figure", dpi=120, facecolor=(1, 1, 1))
    plt.rc("font", family='stixgeneral', size=15)
    plt.rc("axes", titlesize=19)
    plt.rc("axes", facecolor=(1, 1, 1))
    plt.rc("mathtext", fontset='cm')

def lighten(color, factor=2):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - (1 - c[1])/factor, c[2])

def opacify(color, alpha=.5):
  if len(color) == 3:
    return tuple(list(color) + [alpha])
  else:
    return tuple(list(color)[:3] + [color[3]*alpha])

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
