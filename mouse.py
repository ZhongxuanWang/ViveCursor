from pynput.mouse import Button, Controller
from time import sleep

mouse = Controller()

# Read pointer position

s = ''
while True:
    # s = 'The current pointer position is {0}'.format(mouse.position)
    # print(s)
    mouse.move(1, -1)
    sleep(0.1)

# Set pointer position
mouse.position = (2500, 40)
print('Now we have moved it to {0}'.format(
    mouse.position))


# Move pointer relative to current position


# Press and release
mouse.press(Button.left)
mouse.release(Button.left)

# Double click; this is different from pressing and releasing
# twice on macOS
mouse.click(Button.left, 2)

# Scroll two steps down
mouse.scroll(0, 2)
