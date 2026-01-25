import argparse
import cairo
import colorsys
import math
import numpy
import os
import re
import sys

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint

# Given the dimensions of an initial pentagon, creates a vector graphics
# image of a "snake" created by wrapping smaller and smaller similar pentagons.
# The arguments to the program are the filename followed by ten numbers.
# The numbers are the attributes of the pentagon in counterclockwise order:
# angle, side length, angle, side length, angle, side length...
# The angles are specified in degrees.

# However, there is one wrinkle. Ten parameters overspecifies a pentagon.
# One of the angles should be specified as zero; the program will create
# the fifth angle, so that the angles sum to 540.

# Only three of the five side lengths should be specified. The other two
# should be zero. The program will create the correct lengths to make the
# angles work out. It is not possible to construct a pentagon from all
# possible lengths. If the lengths are not valid, the program will produce
# an unhelpful message.

# The program outputs a .svg file to avoid losing fidelity due to pixellation.
# If a .png file is desired, the output can be converted usin Inkscape
# (https://inkscape.org).
# A sample command line might be:
#   "c:\Program Files\Inkscape\inkscape.com" -f=filename.svg \
#   --export-png=filename.png --export-dpi=3000 --export-background=white

# You may have to experiment with the export-dpi option and possibly the
# line_width parameter in the code to get a satisfactory result. A program
# like Adobe Illustrator is a more flexible and convenient, but it isn't free.


def main():
    try:
        args = process_args()
        pentagon_arg = args["pentagon"]
        if pentagon_arg is None:
            pentagon = solve_pentagon(
                args["angle"], args["position"], args["period"]
            )
        else:
            pentagon = gen_pentagon(*pentagon_arg)

        scale_factor = math.dist(pentagon[3], pentagon[4]) / math.dist(
            pentagon[0], pentagon[1]
        )
        if scale_factor >= 1:
            raise Exception(f"{scale_factor=}")

        # The reason for outputting an .svg file is so that we don't
        # have to worry about the scale of the diagram. Except that we
        # kind of do. We must specify the line width before we draw
        # our first line. And at this point, we don't know how big the
        # final result. So we base the line width on the approximate
        # dimensions of the first pentagon and hope for the best.
        pentagon_size = max(
            max(p[0] for p in pentagon) - min(p[0] for p in pentagon),
            max(p[1] for p in pentagon) - min(p[1] for p in pentagon),
        )

        with SVGWriter(
            args["filename"], line_width=0.005 * pentagon_size
        ) as ctx:
            limit = math.dist(pentagon[1], pentagon[2]) / 100
            iterations = 0
            period = args["period"]

            def draw(pent):
                draw_polygon(
                    ctx,
                    pent,
                    (
                        360.0 * (iterations % period) / float(period)
                        if args["color"]
                        else None
                    ),
                )

            draw(pentagon)
            while True:
                scaled = [
                    (v[0] * scale_factor, v[1] * scale_factor) for v in pentagon
                ]
                r = Rigid(scaled[0], scaled[1], pentagon[4], pentagon[3])
                next_pentagon = [r.translate(v) for v in scaled]
                iterations += 1
                draw(next_pentagon)
                pentagon = next_pentagon
                if math.dist(pentagon[1], pentagon[2]) <= limit:
                    break
                if iterations > 200:
                    print("overrun")
                    break

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


def draw_polygon(ctx, vertices, hue):
    if hue is not None:
        ctx.move_to(*vertices[0])
        for v in vertices[1:]:
            ctx.line_to(*v)
        ctx.close_path()
        ctx.save()
        ctx.set_source_rgba(*colorsys.hls_to_rgb(hue / 360.0, 0.6, 1.0), 0.75)
        ctx.fill()
        ctx.restore()

    ctx.move_to(*vertices[0])
    for v in vertices[1:]:
        ctx.line_to(*v)
    ctx.close_path()
    ctx.stroke()


def solve_pentagon(angle, position, period):
    # Cycle of six
    v2 = (period - 2) * 180.0 / period
    angles = [angle, v2, 180 - angle, 180 + angle - v2, 180 - angle]
    print(f"{angles=}")

    theta = 270
    cosz = []
    sinz = []
    for a in angles:
        cosz.append(math.cos(math.radians(theta)))
        sinz.append(math.sin(math.radians(theta)))
        theta = theta + 180 + a
        while theta >= 360:
            theta -= 360

    A = numpy.array(cosz[1:])
    cos_constraint = LinearConstraint(A, -cosz[0], -cosz[0])

    B = numpy.array(sinz[1:])
    sin_constraint = LinearConstraint(B, -sinz[0], -sinz[0])

    def size_fn(x):
        b, c, d, e = x
        return e - (d ** (period - 1)) * (c + b * d)

    size_constraint = NonlinearConstraint(size_fn, 0, 0)

    def mid_fn(x):
        b, c, d, e = x
        return c * (position) - (b * d) * (1.0 - position)

    mid_constraint = NonlinearConstraint(mid_fn, 0, 0)

    solution = minimize(
        lambda x: max(x),
        numpy.array([0.5, 0.5, 0.5, 0.5]),
        constraints=[
            sin_constraint,
            cos_constraint,
            size_constraint,
            mid_constraint,
        ],
    )

    all_sides = [1.0] + solution.x.tolist()
    a, b, c, d, e = all_sides
    print(f"sides={all_sides}")

    v = (0, 0)
    vertices = [v]
    theta = 270
    for i in range(4):
        side = all_sides[i]
        next_v = (
            v[0] + side * math.cos(math.radians(theta)),
            v[1] + side * math.sin(math.radians(theta)),
        )
        theta = theta + 180 + angles[i]
        while theta >= 360:
            theta -= 360
        vertices.append(next_v)
        v = next_v
    return vertices


# A pentagon can be determined by specifying all five angles,
# and three of the five sides. The input has zeroes
# in the sides for the sides to be computed. theta is the angle with
# respect to the x-axis of the first side.
# Returns the Cartesian coordinates of the five vertices.
# The first vertex will be (0, 0)/540
def gen_pentagon(sides, angles):
    theta = 0
    cosines = []
    sines = []
    for a in angles:
        cosines.append(math.cos(math.radians(theta)))
        sines.append(math.sin(math.radians(theta)))
        theta = theta + 180 + a
        while theta >= 360:
            theta -= 360

    p0 = sum(cosines[i] * sides[i] for i in range(5) if sides[i])
    p1 = sum(sines[i] * sides[i] for i in range(5) if sides[i])
    a0, b0 = (cosines[i] for i in range(5) if sides[i] == 0)
    a1, b1 = (sines[i] for i in range(5) if sides[i] == 0)
    D = a0 * b1 - a1 * b0
    if D == 0:
        raise Exception("Determinant is zero")
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
    theta = 270
    for i in range(4):
        next_vertex = (
            v[0] + all_sides[i] * math.cos(math.radians(theta)),
            v[1] + all_sides[i] * math.sin(math.radians(theta)),
        )
        theta = theta + 180 + angles[i]
        while theta >= 360:
            theta -= 360
        vertices.append(next_vertex)
        v = next_vertex
    return vertices


def parse_filename(filename):
    if not filename.endswith(".svg"):
        raise argparse.ArgumentTypeError("Filename should end with .svg")

    directory = os.path.dirname(filename)
    if directory and not os.path.isdir(directory):
        raise argparse.ArgumentTypeError(
            f"{filename} is not in a valid directory"
        )
    return filename


def parse_float(arg):
    if arg == "":
        raise argparse.ArgumentTypeError("missing argument")
    if not re.match(r"\d*(\.\d*)?$", arg):
        raise argparse.ArgumentTypeError(f"{arg} is not a number")
    return float(arg)


def parse_int(arg):
    if arg == "":
        raise argparse.ArgumentTypeError("missing argument")
    if not re.match(r"^\d+$", arg):
        raise argparse.ArgumentTypeError(f"{arg} is not a number")
    return int(arg)


def parse_pentagon(arg):
    values = arg.split(",")
    if len(values) != 10:
        raise argparse.ArgumentTypeError("does not have 10 values")
    nums = []
    for v in values:
        if v == "":
            nums.append(0.0)
        else:
            nums.append(parse_float(v))
    sides = [nums[i] for i in (0, 2, 4, 6, 8)]
    angles = [nums[i] for i in (1, 3, 5, 7, 9)]

    if sides.count(0.0) != 2:
        raise Exception("Two sides should be zero")

    if angles.count(0.0) != 1:
        raise Exception("One angle should be zero")

    if sum(angles) >= 540:
        raise Exception("Sum of angles exceeds 540")

    # Make the angles add up to 540
    angles[angles.index(0.0)] = 540.0 - sum(angles)

    return (sides, angles)


def parse_angle(arg):
    value = parse_float(arg)
    if value == 0.0:
        raise argparse.ArgumentTypeError("Cannot be zero")
    return value


def parse_position(arg):
    value = parse_float(arg)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("Must be in range [0..1]")
    return value


def parse_period(arg):
    value = parse_int(arg)
    if value <= 2:
        raise argparse.ArgumentTypeError("Must be at least 3")
    return value


def is_option(name, arg):
    return name == arg or arg.startswith(f"{name}=")


def process_args():
    parser = argparse.ArgumentParser(
        description="Produces a spiral of pentagons"
    )
    parser.add_argument("filename", type=parse_filename)
    parser.add_argument(
        "--pentagon", type=parse_pentagon, help="the dimensions of the pentagon"
    )
    parser.add_argument(
        "--angle",
        type=parse_angle,
        default=90.0,
        help="key angle of the pentagon",
    )
    parser.add_argument("--position", type=parse_position, default=0.5)
    parser.add_argument(
        "--period",
        type=parse_period,
        default=6,
        help="number of pentagons in one cycle",
    )
    parser.add_argument(
        "--color", action="store_true", help="add colors to the diagram"
    )
    args = vars(parser.parse_args())

    # Horrible hack.
    # It is not easy to have argparse tell us whether an option that has a
    # default was specified.
    if args["pentagon"] is not None:
        for a in sys.argv[1:]:
            if is_option("--angle", a):
                raise Exception("--angle and --pentagon cannot both be present")
            if is_option("--position", a):
                raise Exception(
                    "--position and --pentagon cannot both be present"
                )
            if is_option("--period", a):
                raise Exception(
                    "--period and --pentagon cannot both be present"
                )
    return args


if __name__ == "__main__":
    main()
