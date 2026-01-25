import genetic_utils as gu

p1 = [0 for _ in range(20)]
p2 = [1 for _ in range(20)]

print(p1,p2)

o1, o2 = gu.multi_point_crossover(p1, p2, 3)
print(o1, o2)