import gmsh
import numpy as np
import math

gmsh.initialize()

# load stl file
gmsh.open('mesh.stl')

entities = gmsh.model.getEntities()
totalCoords = []
totalElemNodeTags = []


for e in entities:
    # Dimension and tag of the entity:
    dim = e[0]
    tag = e[1]
    
    nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(dim, tag)
    totalCoords.append(np.reshape(np.array(nodeCoords),np.size(nodeCoords)).tolist())
    
    # Get the mesh elements for the entity (dim, tag):
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim, tag)


totalCoords = np.array(nodeCoords)
totalElemNodeTags = np.array(elemNodeTags)
elemNodeTagMap = np.arange(len(nodeTags))

elemCount = 0

#loop over all triangles
for i in range(np.size(totalElemNodeTags)):
    for j in range(len(nodeTags)):
        if totalElemNodeTags[0,i] == nodeTags[j]:
            totalElemNodeTags[0,i] = j

totalElemNodeTags = np.reshape(totalElemNodeTags,newshape=(int(np.size(totalElemNodeTags)/3),3))# - np.ones((int(np.size(totalElemNodeTags)/3),3))

# generate coordinates
totalCoords = np.reshape(totalCoords,newshape=(int(np.size(totalCoords)/3),3))

cell_centroids = np.zeros((np.shape(totalElemNodeTags)[0],2))
cell_volumes = np.zeros((np.shape(totalElemNodeTags)[0]))
for i in range(np.shape(totalElemNodeTags)[0]):
    node1 = int(totalElemNodeTags[i,0])
    node2 = int(totalElemNodeTags[i,1])
    node3 = int(totalElemNodeTags[i,2])
    
    cx = (totalCoords[node1,0] + totalCoords[node2,0] + totalCoords[node3,0]) / 3
    cy = (totalCoords[node1,1] + totalCoords[node2,1] + totalCoords[node3,1]) / 3
    
    cell_centroids[i,0] = cx
    cell_centroids[i,1] = cy
    
    # get volume
    x1 = totalCoords[node1,0]
    y1 = totalCoords[node1,1]
    x2 = totalCoords[node2,0]
    y2 = totalCoords[node2,1]
    x3 = totalCoords[node3,0]
    y3 = totalCoords[node3,1]
    
    vol = .5*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    cell_volumes[i] = vol


# generate edge index for cells
edge_index = []
edge_centers = []
n_hat = []
faceArea = []
for i in range(np.shape(totalElemNodeTags)[0]):
    node1i = int(totalElemNodeTags[i,0])
    node2i = int(totalElemNodeTags[i,1])
    node3i = int(totalElemNodeTags[i,2])
    nodei = [node1i,node2i,node3i]
    
    for j in range(np.shape(totalElemNodeTags)[0]):
        node1j = int(totalElemNodeTags[j,0])
        node2j = int(totalElemNodeTags[j,1])
        node3j = int(totalElemNodeTags[j,2])
        nodej = [node1j,node2j,node3j]
        
        common = set(nodei).intersection(nodej)
        if len(common)==2:
            # find which node is not in this list and then use it as x3, y3
            not_common = list(set(nodei) - set(common))
            
            common = list(common)
            edge_index.append([i,j])
            xc = (totalCoords[common[0], 0] + totalCoords[common[1], 0]) / 2
            yc = (totalCoords[common[0], 1] + totalCoords[common[1], 1]) / 2 
            edge_centers.append([xc, yc])
            
            # generate normal vector (outward facing)
            node3 = not_common[0]
            x1 = totalCoords[common[0],0]
            y1 = totalCoords[common[0],1]
            x2 = totalCoords[common[1],0]
            y2 = totalCoords[common[1],1]
            x3 = totalCoords[node3,0]
            y3 = totalCoords[node3,1]
            
            iout = (y1-y2) / math.sqrt((y1-y2)**2 + (x2-x1)**2)
            jout = (x2-x1) / math.sqrt((y1-y2)**2 + (x2-x1)**2)
            
            det = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
            sign = np.sign(det)
            
            n_hat.append([iout*sign*-1, jout*sign*-1])
            
            # get area of face
            dx = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
            faceArea.append(dx)
            
            
yTop = 1.0
yBottom = 0.0
xLeft = 0.0
xRight = 0.5

xLeftCyl = .5
yLeftCyl = .5
rCyl = .15


leftBounds = []
rightBounds = []
topBounds = []
bottomBounds = []
leftCylBounds = []
topCylBounds = []
bottomCylBounds = []

for i in range(np.shape(totalCoords)[0]):
    xi = totalCoords[i,0]
    yi = totalCoords[i,1]
    
    # check boundaries
    if abs(xi-xLeft) <= 1e-6:
        leftBounds.append(i)
    if abs(xi-xRight) <= 1e-6:
        rightBounds.append(i)
    if abs(yi-yTop) <= 1e-6:
        topBounds.append(i)
    if abs(yi-yBottom) <= 1e-6:
        bottomBounds.append(i)
	
    # check cylinder
    if math.sqrt((xi-xLeftCyl)**2+(yi-yLeftCyl)**2)-rCyl <= 1e-6:
        leftCylBounds.append(i)
	


for i in range(np.shape(totalElemNodeTags)[0]):
    node1i = int(totalElemNodeTags[i,0])
    node2i = int(totalElemNodeTags[i,1])
    node3i = int(totalElemNodeTags[i,2])
    nodei = [node1i, node2i, node3i]
    
    # Left boundary (inflow specified by full primitive state eq, indicate by -1 index)
    intersect = set(nodei).intersection(leftBounds)
    if len(intersect)==2:
        intersect = list(intersect)
        edge_index.append([i,-1])
        y0 = totalCoords[intersect[0],1]
        y1 = totalCoords[intersect[1],1]
        dx = np.abs(y0-y1)
        faceArea.append(dx)
        n_hat.append([-1, 0])
    
    # Right boundary (outflow specified by mass flow rate? (mirror pressure, mirror density, get velocity by specifying mass flow rate, indicate by -2 index)
    intersect = set(nodei).intersection(rightBounds)
    if len(intersect)==2:
        intersect = list(intersect)
        edge_index.append([i,-2])
        y0 = totalCoords[intersect[0],1]
        y1 = totalCoords[intersect[1],1]
        dx = np.abs(y0-y1)
        faceArea.append(dx)
        n_hat.append([1, 0])
        
                
    # Top boundary (slip condition, indicate by -3 index)
    intersect = set(nodei).intersection(topBounds)
    if len(intersect)==2:
        intersect = list(intersect)
        edge_index.append([i,-3])
        x0 = totalCoords[intersect[0],0]
        x1 = totalCoords[intersect[1],0]
        dx = np.abs(x0-x1)
        faceArea.append(dx)
        n_hat.append([0, 1])
    
    # Bottom boundary (slip condition, indicate by -3 index)
    intersect = set(nodei).intersection(bottomBounds)
    if len(intersect)==2:
        intersect = list(intersect)
        edge_index.append([i,-3])
        x0 = totalCoords[intersect[0],0]
        x1 = totalCoords[intersect[1],0]
        dx = np.abs(x0-x1)
        faceArea.append(dx)
        n_hat.append([0, -1])
    
	
	# left cylinder (likely no slip condition, indicated by a -4 in the edge index)
    common = set(nodei).intersection(leftCylBounds)
    # if two vertices of the triangle are on the boundary, then this is a boundary cell
    if len(common)==2:
        common = list(common)
        edge_index.append([i,-4])
        not_common = list(set(nodei) - set(common))
            
        common = list(common)
        xc = (totalCoords[common[0], 0] + totalCoords[common[1], 0]) / 2
        yc = (totalCoords[common[0], 1] + totalCoords[common[1], 1]) / 2 
        edge_centers.append([xc, yc])
            
        # generate normal vector (outward facing)
        node3 = not_common[0]
        x1 = totalCoords[common[0],0]
        y1 = totalCoords[common[0],1]
        x2 = totalCoords[common[1],0]
        y2 = totalCoords[common[1],1]
        x3 = totalCoords[node3,0]
        y3 = totalCoords[node3,1]
            
        iout = (y1-y2) / math.sqrt((y1-y2)**2 + (x2-x1)**2)
        jout = (x2-x1) / math.sqrt((y1-y2)**2 + (x2-x1)**2)
            
        det = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
        sign = np.sign(det)
            
        n_hat.append([iout*sign*-1, jout*sign*-1])
            
        # get area of face
        dx = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
        faceArea.append(dx)
        


np.save('Mesh/edge_index', np.array(edge_index))
np.save('Mesh/cell_centroids', cell_centroids)
np.save('Mesh/n_hat', np.array(n_hat))
np.save('Mesh/faceArea', np.array(faceArea))
np.save('Mesh/cell_volumes', cell_volumes)

gmsh.finalize()