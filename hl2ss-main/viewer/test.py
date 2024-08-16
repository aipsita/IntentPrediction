from scipy.ndimage import uniform_filter

head_positions = [[1,1,1],
                 [2,2,2],
                 [3,3,30],
                 [1,1,1],
                 [2,2,2],
                 [3,3,3]]
# Smooth positions to reduce noise
head_smoothed_positions = uniform_filter(head_positions, size=3)
print(head_smoothed_positions)