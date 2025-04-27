from scipy.stats import norm
from csv import writer


distribution_x = norm(loc=0, scale=100)
distribution_y = norm(loc=0, scale=100)
distribution_z = norm(loc=0.2, scale=0.05)

num_points = 1000
x = distribution_x.rvs(size=num_points)
y = distribution_y.rvs(size=num_points)
z = distribution_z.rvs(size=num_points)

points = zip(x, y, z)
with open('Lab1DataPozioma.xyz', 'w', encoding='utf-8', newline='\n') as csvfile:
    csvwriter = writer(csvfile)
    for p in points:
        csvwriter.writerow(p)
