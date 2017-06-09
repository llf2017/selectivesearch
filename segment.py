import numpy as np
from skimage import io, filters

'''
    'Efficient Graph-Based Image Segmentation' By PEDRO F. FELZENSZWALB
'''

def load_image(filename):
    return io.imread(filename)
    
class Edge:
    def __init__(self, start, end, weight):
        self.start = start
        self.end = end
        self.weight = weight
    
    def __lt__(self, other):
        return self.weight < other.weight
        
class Element:
    def __init__(self, p):
        self.parent = p
        self.rank = 0
        self.size = 1
    
class Universe:
    def __init__(self, num_elements):
        self.num = num_elements
        self.elements = list(map(lambda i:Element(i), range(num_elements)))
        
    def find(self, x):
        y = x
        while y != self.elements[y].parent:
            y = self.elements[y].parent
        self.elements[x].parent = y
        return y
        
    def join(self, x, y):
        if self.elements[x].rank > self.elements[y].rank:
            self.elements[y].parent = x
            self.elements[x].size += self.elements[y].size
        else:
            self.elements[x].parent = y
            self.elements[y].size += self.elements[x].size
            if self.elements[x].rank == self.elements[y].rank:
                self.elements[y].rank += 1
        self.num -= 1
        
    def size(self, x):
        return self.elements[x].size
        
def _threshold(c, size):
    return c / size
    
def segment_graph(num_vertices, edges, c):
    ''' segment a graph '''
    ''' return a disjoing-set forest representing the segmentation '''
    
    edges = sorted(edges)
    
    universe = Universe(num_vertices)
    threshold = [c] * num_vertices
    
    for edge in edges:
        start = universe.find(edge.start)
        end = universe.find(edge.end)
        
        if start != end:
            if edge.weight <= min(threshold[start], threshold[end]):
                universe.join(start, end)
                start = universe.find(start)
                threshold[start] = edge.weight + _threshold(c, universe.size(start))
                
    return universe
    
def _gradient(smooth_img, start, end):
    diff = smooth_img[start] - smooth_img[end]
    return np.sqrt(np.sum(np.square(diff), axis=1))
    
def _smooth(img, sigma):
    #tmp = filters.gaussian(img, sigma=sigma, multichannel=True)
    return filters.gaussian(img, sigma=sigma, multichannel=True)
    
def _build_vertices(h, w):
    #(x, y), (x + 1, y)
    j = np.array(list(range(h)))
    j = np.expand_dims(j, 1)
    j = np.tile(j, (1, w-1))
    j = j.reshape((-1))
    
    i = np.array(list(range(w-1)))
    i = np.expand_dims(i, 0)
    i = np.tile(i, (h, 1))
    i = i.reshape((-1))
    
    s1 = j * w + i
    e1 = s1 + 1
    
    #(x, y), (x, y + 1)
    j = np.array(list(range(h-1)))
    j = np.expand_dims(j, 1)
    j = np.tile(j, (1, w))
    j = j.reshape((-1))
    
    i = np.array(list(range(w)))
    i = np.expand_dims(i, 0)
    i = np.tile(i, (h-1, 1))
    i = i.reshape((-1))
    
    s2 = j * w + i
    e2 = s2 + w
    
    #(x, y), (x + 1, y + 1)
    j = np.array(list(range(h-1)))
    j = np.expand_dims(j, 1)
    j = np.tile(j, (1, w-1))
    j = j.reshape((-1))
    
    i = np.array(list(range(w-1)))
    i = np.expand_dims(i, 0)
    i = np.tile(i, (h-1, 1))
    i = i.reshape((-1))
    
    s3 = j * w + i
    e3 = s3 + w + 1
    
    #(x, y), (x + 1, y - 1)
    j = np.array(list(range(1, h)))
    j = np.expand_dims(j, 1)
    j = np.tile(j, (1, w-1))
    j = j.reshape((-1))
    
    i = np.array(list(range(w-1)))
    i = np.expand_dims(i, 0)
    i = np.tile(i, (h-1, 1))
    i = i.reshape((-1))
    
    s4 = j * w + i
    e4 = s4 - w + 1
    
    # all valid
    s = np.concatenate((s1, s2, s3, s4))
    e = np.concatenate((e1, e2, e3, e4))
    
    return s, e
    
def segment_image(img, sigma, scale, min_size):
    scale = float(scale) / 255
    
    height, width = img.shape[:2]
    smooth_img = _smooth(img, sigma) # this will convert to float [0-1]
    start, end = _build_vertices(height, width)
    diff = _gradient(np.reshape(smooth_img, (-1, 3)), start, end)
    
    # build graph
    edges = list(map(lambda s, e, d:Edge(s, e, d), start, end, diff))
    
    # segment
    universe = segment_graph(height*width, edges, scale)
    
    # post process small components
    for edge in edges:
        start = universe.find(edge.start)
        end = universe.find(edge.end)
        
        if start != end and min(universe.size(start), universe.size(end)) < min_size:
            universe.join(start, end)
            
    # number of connected components in the segmentation.
    num_css = universe.num
    
    colors = np.random.randint(256, size=(width*height, 3))
    output = np.zeros_like(img)
    segment_mask = np.zeros((height, width))
    comp_map = {}
    cur_index = 0
    
    for y in range(height):
        for x in range(width):
            comp = universe.find(y * width + x)
            output[y, x] = colors[comp]
            if comp in comp_map:
                segment_mask[y, x] = comp_map[comp]
            else:
                segment_mask[y, x] = cur_index
                comp_map[comp] = cur_index
                cur_index += 1
            
    return output, num_css, segment_mask
    
if __name__ == '__main__':
    from skimage import data
    import matplotlib.pyplot as plt
    
    cat = data.chelsea()
    
    output, num_css, segment_mask = segment_image(cat, 0.8, 255, 20)
    print('number of connected components in the segmentation: ', num_css)
    
    plt.subplot(1, 2, 1)
    plt.imshow(cat)
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.show()
