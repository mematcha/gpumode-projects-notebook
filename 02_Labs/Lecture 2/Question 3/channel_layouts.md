### Channel Layouts: Channel-First vs Channel-Last

Channel layout determines how image data is arranged in memory. In **channel-first (CHW)** format, all pixels for one channel are stored together, followed by the next channel, while in **channel-last (HWC)** format, the RGB values for each pixel are stored next to each other. For an image of size H \times W with 3 channels, both layouts store exactly 3HW values, but the memory indexing differs. In CHW, the index is `channel * H * W + row * W + col`, whereas in HWC it is `(row * W + col) * 3 + channel`.

Channel First: RRRRRR GGGGGG BBBBBB

Channel Last: RGB RGB RGB RGB RGB RGB

**channel-first = grouped by channel**, **channel-last = grouped by pixel**.