import pandas as pd

# Load the CSV file
data = pd.read_csv('C:/Users/anany/Downloads/Feature/Data-1/User6/Study2/20240708_151337797-data.csv')

# Find the indices where value in column 'a' is 1
indices_a_is_1 = data[data['KeyPress'] == '1'].index

# Find the indices where value in column 'b' is equal to the value of the previous row in column 'b'
indices_b_equal_previous = data[data['TargetName'] == data['TargetName'].shift(1)].index

# Find the common indices between the two conditions
common_indices = indices_a_is_1.intersection(indices_b_equal_previous).astype(pd.Int64Dtype())

# Create a new column 'Key Press Number' and populate it with the respective indices
data.loc[indices_a_is_1, 'Key Press Number'] = range(1, len(indices_a_is_1) + 1)

# Print the common indices
print(common_indices.tolist())

print(data.loc[common_indices, 'Key Press Number'].astype(pd.Int64Dtype()).tolist())

#User 1:
#Study 1:
#[615, 2171, 2390, 2688, 3235, 3282, 3324, 3581, 3827]
#[10, 46, 49, 55, 67, 69, 70, 76, 81]
#Study 2:
#[125, 437, 519, 553, 798, 1022, 1831, 2226, 2526, 2879]
#[3, 11, 12, 13, 21, 28, 56, 67, 71, 82]

#User 2:
#Study 1:
#[1999]
#[73]
#Study 2:
#[525, 837, 2291]
#[14, 19, 75]

#User 3:
#Study 1:
#[1984, 2135]
#[66, 74]
#Study2:
#[308, 1304, 1487, 1565]
#[11, 61, 72, 76]

#User 4:
#Study 1:
#[602, 837, 1976]
#[18, 21, 75]
#Study 2:
#[837, 1059, 1290, 1473]
#[39, 52, 65, 76]

#User 5:
#Study 1:
#[696, 771, 2554, 2658]
#[1, 2, 71, 76]
#Study 2:
#[70, 1698]
#[1, 74]

#User 6:
#Study 1:
#[1598, 1719]
#[67, 74]
#Study 2:
#[250, 1512]
#[10, 74]