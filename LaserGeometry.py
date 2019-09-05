import turtle,math,random,pdb

TAU = 2 * math.pi

def equals(a, b, epsilon = 10 ** -2):
    return abs(a - b) < epsilon

class Vector:
    def __init__(self, *components):
        self.components = components
    def from_angle(theta, magnitude = 1):
        return Vector(math.cos(theta) * magnitude, math.sin(theta) * magnitude)
    def draw(self):
        turtle.penup()
        turtle.goto(self.components)
        turtle.dot(3)
    def indices(self):
        return range(len(self))
    def zip_map(self, other, func):
        return [func(self[i], other[i]) for i in self.indices()]
    def __str__(self):
        if len(self) == 2:
            return "{}i + {}j".format(*self.components)
        else:
            return str(self.components)
    def __getitem__(self, index):
        return self.components[index]
    def __iter__(self):
        return iter(self.components)
    def __len__(self):
        return len(self.components)
    def __add__(self, other):
        return Vector(*[self[i] + other[i] for i in self.indices()])
    def __sub__(self, other):
        return Vector(*[self[i] - other[i] for i in self.indices()])
    def __neg__(self):
        return self.mul(-1)
    def mul(self, scalar):
        return Vector(*map(lambda x: x*scalar, self.components))
    def div(self, scalar):
        return Vector(*map(lambda x: x/scalar, self.components))
    def to_theta(self):
        return math.atan2(self[1], self[0])
    def magnitude(self):
        return math.sqrt(sum([a * a for a in self]))
    def dot(self, other):
        return sum([self[i] * other[i] for i in self.indices()])

class Ray:
    def __init__(self, start, direc):
        self.start = start
        self.direc = direc
    def __str__(self):
        return "Start: {}, Direction: {}".format(self.start, self.direc)
    def point_on(self, u):
        return Vector(self.start[0] + self.direc[0] * u, self.start[1] + self.direc[1] * u)
    def intersect(self, other):
        det = other.direc[0] * self.direc[1] - other.direc[1] * self.direc[0]
        if det == 0: return None
        dx = other.start[0] - self.start[0]
        dy = other.start[1] - self.start[1]
        u = (dy * other.direc[0] - dx * other.direc[1]) / det
        v = (dy * self.direc[0] - dx * self.direc[1]) / det
        if u < 0 or v < 0: return None
       # pdb.set_trace()
        return (self.start[0] + self.direc[0] * u, self.start[1] + self.direc[1] * u)
    def reflect(self, normal, point):
        direc = self.direc - normal.mul(2 * self.direc.dot(normal))
        return Ray(point, direc)
    def draw(self, limit = None):
        if limit == None: limit = Vector(*turtle.screensize()).magnitude()
        turtle.goto(self.start)
        turtle.dot(4)
        turtle.pendown()
        turtle.radians()
        turtle.setheading(self.direc.to_theta())
        turtle.fd(limit)

class Segment:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __str__(self):
        return "{} to {}".format(self.start, self.end)
    def intersect(self, ray):
       # pdb.set_trace()
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        det = ray.direc[1]*dx - ray.direc[0]*dy
        if det == 0: return None
        v = (ray.direc[1] * (ray.start[0] - self.start[0]) - ray.direc[0] * (ray.start[1] - self.start[1])) / det
        if 0 > v or v > 1: return None
        u = None
        try:
            u = (self.start[0] + dx * v - ray.start[0]) / ray.direc[0]
        except ZeroDivisionError:
            try:
                u = (self.start[1] + dy * v - ray.start[1]) / ray.direc[1]
            except ZeroDivisionError:
                raise ValueError("invalid ray direction vector: {}".format(ray))
        if u < 0 or equals(u, 0): return None
        return (self.start[0] + dx * v, self.start[1] + dy * v)
    def pos_vec(self):
        return Vector(*self.end) - Vector(*self.start)
    def normal(self):
        return Vector.from_angle(self.pos_vec().to_theta() + TAU / 4)
    def draw(self):
        turtle.goto(self.start)
        turtle.pendown()
        turtle.goto(self.end)

class Rect:
    def __init__(self, x0, x1, y0, y1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
    def all_corners(self):
        yield (self.x0,self.y0)
        yield (self.x1,self.y0)
        yield (self.x1,self.y1)
        yield (self.x0,self.y1)
    def all_sides(self):
        point = self.all_corners()
        p_start = next(point)
        p_prev = p_start
        for p_next in point:
            yield Segment(p_prev, p_next)
            p_prev = p_next
        yield Segment(p_prev, p_start)
    def intersect(self, ray):
        min_dist = None
        point = None
        for side in self.all_sides():
       #     pdb.set_trace()
            p = side.intersect(ray)
            if p == None: continue
            dist = (Vector(*p) - ray.start).magnitude()
            if min_dist == None or dist < min_dist:
                min_dist = dist
                point = p
        return point
    def get_side(self, point):
        for side in self.all_sides:
            if side.contains(point):
                yield side
    def normal_at_corner(self, point):
        # the bottom left and upper right corners have normals at 45 degrees
        if (equals(point[0], self.x0) and equals(point[1], self.y0)) or (equals(point[0], self.x1) and equals(point[1], self.y1)):
            return Vector.from_angle(TAU/8)
        # the bottom right and upper left corners have normals at 135 degrees
        if (equals(point[0], self.x1) and equals(point[1], self.y0)) or (equals(point[0], self.x0) and equals(point[1], self.y1)):
            return Vector.from_angle(3*TAU/8)
        return None
    def reflect_ray(self, ray):
        # return the ray's reflection on this object, the point of reflection, and the distance of the point to the start of the ray
        temp = []
        for side in self.all_sides():
            pnt = side.intersect(ray)
            if pnt == None: continue
            dist = (Vector(*pnt) - ray.start).magnitude()
            if equals(0, dist): continue
            temp.append((pnt,side,dist))
        if len(temp) == 0: return None, None, None
        pnt, side, dist = min(temp, key = lambda pnt_side_dist: pnt_side_dist[2])
        # normals have slightly different behavior at the side
        normal = self.normal_at_corner(pnt)
        if normal == None:
            normal = side.normal()
        reflect = ray.reflect(normal, pnt)
        return reflect, pnt, dist
    def draw(self):
        turtle.goto(self.x0,self.y1)
        turtle.pendown()
        for point in self.all_corners():
            turtle.goto(point)

class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    def intersect(self, ray, epsilon = 2):
        # return where the ray intersects this
        dx = ray.start[0] - self.center[0]
        dy = ray.start[1] - self.center[1]
        # use quadratic formula
        A = ray.direc[0]**2 + ray.direc[1]**2
        B = 2*dx*ray.direc[0] + 2*dy*ray.direc[1]
        C = dx*dx + dy*dy - self.radius*self.radius
        discr = B*B - 4 * A * C
        if discr < 0: return None # no solutions
        if discr == 0:
            u = -B / 2 / A # exactly one solution
        else:
            # 2 possible candidates
            u1 = (-B + math.sqrt(discr)) / 2 / A
            u2 = (-B - math.sqrt(discr)) / 2 / A
            # negative or below a certain threshold are rejected
            # otherwise choose the smallest one
            if u1 < epsilon: u = u2
            elif u2 < epsilon: u = u1
            else: u = min(u1,u2)
        if u < epsilon: return None
        return ray.point_on(u)
    def reflect_ray(self, ray):
        # return the ray's reflection on this object, the point of reflection, and the distance of the point to the start of the ray
        pnt = self.intersect(ray)
        if pnt == None: return None, None, None
        normal = Vector(*pnt) - self.center
        normal = normal.div(normal.magnitude())
        reflect = ray.reflect(normal, pnt)
        dist = (Vector(*pnt) - ray.start).magnitude()
        return reflect, pnt, dist
    def draw(self):
        turtle.setheading(0)
        turtle.goto(self.center[0], self.center[1] - self.radius)
        turtle.pendown()
        turtle.circle(self.radius)

class MultiLine:
    def __init__(self, points, step = None):
        self.points = points
        self.step = step
                    
        self.look_up = {}
        for index, point in enumerate(points):
            key = self.get_key(point)
            try:
                self.look_up[key].append(index)
            except KeyError:
                self.look_up[key] = [index]
    def get_key(self, point):
        return ((point[0] // self.step) * self.step, (point[1] // self.step) * self.step)
    def get_nearest_point(self, point):
        index = None
        min_dist = None
        key = self.get_key(point)
        for dx in [-self.step, 0, self.step]:
            for dy in [-self.step, 0, self.step]:
                nkey = (key[0] + dx, key[1] + dy)
                for candidate_index in self.look_up[nkey]:
                    dist = (point - self.points[candidate_index]).magnitude()
                    if min_dist == None or dist < min_dist:
                        min_dist = dist
                        ans_point = candidate_index
        return candidate
    def intersect(self, ray):
        prev = self.points[0]
        for point in self.points:
            pass
    def reflect_ray(self, ray):
        pass
    def draw(self):
        turtle.penup()
        turtle.goto(self.points[0])
        turtle.pendown()
        for point in self.points:
            turtle.goto(point)
def rand_point(dampen = 1):
    w,h = turtle.screensize()
    x = w//2
    y = h//2
    return (random.randint(-x, x) * dampen, random.randint(-y,y) * dampen)
def rand_dir():
    return Vector.from_angle(TAU * random.random())
def rand_ray():
    return Ray(rand_point(dampen = 0.333), rand_dir())
def rand_rect():
    x0,y0 = rand_point()
    x1,y1 = rand_point()
    if x0 > x1: x1,x0 = x0,x1
    if y0 > y1: y1,y0 = y0,y1
    return Rect(x0,x1,y0,y1)
def border_rect():
    w,h = turtle.screensize()
    x = w
    y = h
    return Rect(-x, x, -y, y)
def rand_circ():
    return Circle(rand_point(), random.randint(10,100))

def rand_obj():
    if random.randint(0,1) == 0:
        return rand_rect()
    else:
        return rand_circ()

def draw_all(*drawables, colors = None):
    for drawable in drawables:
        turtle.penup()
        if colors != None:
            turtle.pencolor(*next(colors))
        drawable.draw()

class World:
    def create_random(obj_count = 3):
        objs = [rand_obj() for i in range(obj_count)]
        objs.append(border_rect())
        return World(rand_ray(), objs)
    def __init__(self, laser_ray_initial, objects, calculate = True):
        self.objects = objects
        self.laser_ray = laser_ray_initial
        self.points = [laser_ray_initial.start]
        self.calculated = False
        self.calculate()
    def bounce(self):
        min_dist, new_ray, new_pnt = None, None, None
        # for low numbers of points, it's fine to iterate over them
        # instead implemented some hash-map like data structure
        for thing in self.objects:
            reflect, pnt, dist = thing.reflect_ray(self.laser_ray)
            if reflect != None and (min_dist == None or dist < min_dist):
                new_ray = reflect
                min_dist = dist
                new_pnt = pnt
        return new_ray, new_pnt
    def calculate(self, limit = 1000):
        if self.calculated: return None
        i = 0
        self.free_end = True
        new_ray, new_pnt = self.bounce()
        while new_ray != None and new_pnt != None:
            self.points.append(new_pnt)
            self.laser_ray = new_ray
            new_ray, new_pnt = self.bounce()
            i += 1
            if i >= limit:
                self.free_end = False
                break
        self.calculated = True
    def draw(self):
        self.calculate()
        turtle.pensize(width = 3)
        draw_all(*self.objects)
        turtle.penup()
        turtle.pensize(width = 1)
        for pnt in self.points:
            turtle.goto(pnt)
            turtle.pendown()
        if self.free_end:
           draw_all(self.laser_ray)


w = World.create_random()
w.draw()
#reflect, pnt, dist = rect.reflect_ray(ray)
