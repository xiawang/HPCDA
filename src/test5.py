from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = [1,2,3,4,5,6,7,8,9,10]
Y = [5,6,2,3,13,4,1,2,4,8]
Z = [2,3,3,3,5,7,9,11,9,10]

label = [1,1,1,1,1,0,0,0,0,0]
for i in xrange(10):
	if label[i]==1:
		ax.scatter(X[i],Y[i],Z[i], c='r', marker='o')
	else:
		ax.scatter(X[i],Y[i],Z[i], c='b', marker='o')


ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')

plt.show()