import cairo
import math
import numpy
import os
import re
import sys

# Given the dimensions of an initial pentagon, creates a vector graphics
# image of a "snake" created by wrapping smaller and smaller similar pentagons.
# The arguments to the program are the filename followed by ten numbers.
# The numbers are the attributes of the pentagon in clockwise order:
# angle, side length, angle, side length, angle, side length...
# The angles are specified in degrees. 

# However, there is one wrinkle. Ten parameters overspecifies a pentagon.
# One of the angles should be specified as zero; the program will create
# the fifth angle, so that the angles sum to 540.

# Only three of the five side lengths should be specified. The other two
# should be zero. The program will create the correct lengths to make the
# angles work out. It is not possible to construct a pentagon from all
# specified lengths. If the lengths are not valid, the program will produce
# a unhelpful message.

# The program outputs a .svg file to avoid losing fidelity due to pixellation.
# If a .png file is desired, the output can be converted usin Inkscape
# (https://inkscape.org).
# A sample command line might be:
#   "c:\Program Files\Inkscape\inkscape.com" -f=filename.svg \
#   --export-png=filename.png --export-dpi=3000 --export-background=white

# You may have to experiment with the export dpi and possibly the line_width
# parameter in the code to get a satisfactory result. A program
# like Adobe Illustrator is a more flexible and convenient, but it isn't free. 

def main():
    try:
        make_spiral(*process_args())
    except Exception as e:
        sys.exit(e)


# Rigid defines a distance and angle preserving transformation from
# an (x, y) coordinate to another. The transformation is defined by
# four points. Ideally math.dist(a1, a2) == math.dist(b1, b2). Then the
# transformation maps a1 to b1 and a2 to b2.
# However, because of floating point, we can't guarantee that
# math.dist(a1, a2) == math.dist(b1, b2). What happens in reality is
# that a1 maps to b1, and a2 maps to a point on the half-line from b1 to b2.
class Rigid:
    def __init__(self, a1, a2, b1, b2):
        # the vector from a1 to a2
        a_vector = tuple(y - x for x, y in zip(a1, a2))

        # the vector from b1 to b2
        b_vector = tuple(y - x for x, y in zip(b1, b2))

        # The angle between the two vectors
        theta = math.atan2(*b_vector) - math.atan2(*a_vector)

        # The rotation matrix
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        self._rot_matrix = numpy.array(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
        )

        self._trans = tuple(x - y for x, y in zip(b1, self.rotate(*a1)))

    def rotate(self, x, y):
        point = numpy.array([x, y])
        rotated = numpy.matmul(point, self._rot_matrix)
        return (rotated[0].item(), rotated[1].item())

    def translate(self, point):
        return tuple(e + f for e, f in zip(self.rotate(*point), self._trans))


# Context manager for creating an svg file.  Parameters are filename,
# and the width of the lines. The line color is initialized to opaque black.
# Returns a Cairo context on which to draw.
class SVGWriter:
    def __init__(self, filename, line_width):
        self.filename = filename
        self.rs = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
        self.ctx = cairo.Context(self.rs)

        # Convert Cairo coordinates to tradional x,y coordinates.
        self.ctx.scale(1, -1)

        self.ctx.set_line_width(line_width)
        self.ctx.set_source_rgba(0, 0, 0, 1)

    def __enter__(self):
        return self.ctx

    def __exit__(self, exc_type, exc_value, traceback):
        x, y, width, height = self.rs.ink_extents()
        surface = cairo.SVGSurface(self.filename, width, height)
        context = cairo.Context(surface)
        context.set_source_surface(self.rs, -x, -y)
        context.paint()
        surface.flush()
        surface.finish()
        del context
        del self.ctx
        self.rs.finish()


def draw_polygon(ctx, vertices):
    ctx.move_to(*vertices[0])
    for v in vertices[1:]:
        ctx.line_to(*v)
    ctx.close_path()
    ctx.stroke()


# A pentagon can be determined by specifying all five angles,
# and three of the five sides. The input has zeroes
# in the sides for the sides to be computed. theta is the angle with
# respect to the x-axis of the first side.
# Returns the Cartesian coordinates of the five vertices.
# The first vertex will be (0, 0)/540
def gen_pentagon(sides, angles):
    start_theta = 0
    theta = start_theta
    cosines = []
    sines = []
    for a in angles:
        cosines.append(math.cos(math.radians(theta)))
        sines.append(math.sin(math.radians(theta)))
        theta = theta + 180 - a

    p0 = sum(cosines[i] * sides[i] for i in range(5) if sides[i])
    p1 = sum(sines[i] * sides[i] for i in range(5) if sides[i])
    a0, b0 = (cosines[i] for i in range(5) if sides[i] == 0)
    a1, b1 = (sines[i] for i in range(5) if sides[i] == 0)
    D = a0 * b1 - a1 * b0
    if D <= 0:
        raise Exception("Determinant {D} is not positive")
    missing_sides = ((-p0 * b1 + p1 * b0) / D, (-a0 * p1 + a1 * p0) / D)
    if not all(s > 0 for s in missing_sides):
        raise Exception("A computed side is not positive")

    # Replace the missing sides with the values computed above.
    all_sides = sides
    i = 0
    for j in range(len(all_sides)):
        if all_sides[j] == 0:
            all_sides[j] = missing_sides[i]
            i += 1
    if i != 2:
        raise Exception("Input sides do not contain two zeroes")

    # Now that we have all the sides, we can compute the Cartesian
    # coordinates of the five vertices.
    v = (0, 0)
    vertices = [v]
    theta = start_theta
    for i in range(4):
        next_vertex = (
            v[0] + all_sides[i] * math.cos(math.radians(theta)),
            v[1] + all_sides[i] * math.sin(math.radians(theta)),
        )
        theta = theta + 180 - angles[i]
        vertices.append(next_vertex)
        v = next_vertex

    return vertices


def make_spiral(filename, sides, angles):
    with SVGWriter(filename, line_width=0.1) as ctx:
        pentagon = gen_pentagon(sides, angles)
        limit = math.dist(pentagon[1], pentagon[2]) / 100
        draw_polygon(ctx, pentagon)
        scale_factor = sides[3] / sides[1]
        while True:
            scaled = [
                (v[0] * scale_factor, v[1] * scale_factor) for v in pentagon
            ]
            r = Rigid(scaled[1], scaled[2], pentagon[4], pentagon[3])
            next_pentagon = [r.translate(v) for v in scaled]
            draw_polygon(ctx, next_pentagon)
            pentagon = next_pentagon
            if math.dist(pentagon[1], pentagon[2]) <= limit:
                break


def process_args():
    if len(sys.argv) != 12:
        sys.exit("Expected 11 arguments")
    filename = sys.argv[1]
    if not filename.endswith(".svg"):
        raise Exception("Filename should end with .svg")

    directory = os.path.dirname(filename)
    if directory and not os.path.isdir(directory):
        raise Exception(f"{filename} is not in a valid directory")

    args = []
    for s in sys.argv[2:]:
        if not re.match(r"^\d+(\.\d+)?$", s):
            raise Exception(f"{s} is not a number")
        args.append(float(s))
    sides = [args[i] for i in (0, 2, 4, 6, 8)]
    angles = [args[i] for i in (1, 3, 5, 7, 9)]

    if sides.count(0) != 2:
        raise Exception("Two sides should be zero")

    if angles.count(0) != 1:
        raise Exception("One angle should be zero")

    if sum(angles) >= 540:
        raise Exception("Sum of angles exceeds 540")

    # Make the angles add up to 540
    angles[angles.index(0)] = 540.0 - sum(angles)

    return (filename, sides, angles)


if __name__ == "__main__":
    main()
