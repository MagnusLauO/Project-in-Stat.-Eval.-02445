import csv

genders = ['M', 'F']
ages = [25, 30, 35, 40, 45]
occupation_categories = [1, 2, 3, 4, 5]

rows = []

# Loop: gender → occupation → age → repeat 4 times
for gender in genders:
    for occupation in occupation_categories:
        for age in ages:
            for _ in range(4):
                rows.append([gender, age, occupation, '', ''])

# Sanity check
print(f"Total rows: {len(rows)}")  # Should be 200

# Write to CSV
with open('people_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['gender', 'age', 'occupation_category', 'low_salary', 'high_salary'])  # header
    writer.writerows(rows)
