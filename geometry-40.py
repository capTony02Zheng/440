# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
from alien import Alien
from typing import List, Tuple
from copy import deepcopy


def within(a,b,c):
    return a >= min(b,c) and a <= max(b,c)



def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """

    # get_centroid()##: Returns the centroid position of the alien (x,y)
    # get_head_and_tail()##: Returns a list with the (x,y) coordinates of the alien's head and tail [(x_head,y_head),(x_tail,y_tail)], which are coincidental if the alien is in its disk form.
    # get_length()##: Returns the length of the line segment of the current alien shape
    # get_width()##: Returns the radius of the current shape. In the ball form this is just simply the radius. In the oblong form, this is half of the width of the oblong, which defines the distance "d" for the sausage shape.
    # is_circle()##: Returns whether the alien is in circle or oblong form. True if alien is in circle form, False if oblong form.
    a = alien
    circle = a.is_circle()
    cen = a.get_centroid()
    headtail = None
    if(not circle):
        headtail = a.get_head_and_tail()
    wid = a.get_width()
    if(circle):
        # print("Hello")
        # print(cen)
        # print(wid)
        for line in walls:
            # print(line)
            l = tuple([(line[0],line[1]),(line[2],line[3])])
            if(point_segment_distance(cen,l) <= wid):
                # print(point_segment_distance(cen,l),wid)
                return True
    else:
        # print("world")
        # print(headtail)
        for line in walls:
            l = tuple([(line[0],line[1]),(line[2],line[3])])
            # print(l)
            head = headtail[0]
            tail = headtail[1]
            ver = tuple([(head[0],head[1]+wid),(tail[0],tail[1]-wid)])
            hor = tuple([(head[0]+wid,head[1]),(tail[0]-wid,tail[1])])
            if( segment_distance(l,headtail) < wid) :
                return True
    
    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
    x0=0
    x1=window[0]
    y0=0
    y1=window[1]
    p1 = [x0,y0]
    p2 = [x0,y1]
    p3 = [x1,y1]
    p4 = [x1,y0]
    poly = [p1,p2,p3,p4]


    a = alien
    circle = a.is_circle()
    cen = a.get_centroid()
    headtail = (0,0)

    if(not circle):
        headtail = a.get_head_and_tail()
    wid = a.get_width()
    if(circle):
        # print("hello")
        # print(cen)
        # print(wid)
        # print(poly)
        f1 = is_point_in_polygon(tuple([cen[0],cen[1]+wid]),poly)
        f2 = is_point_in_polygon(tuple([cen[0],cen[1]-wid]),poly)
        f3 = is_point_in_polygon(tuple([cen[0]-wid,cen[1]]),poly)
        f4 = is_point_in_polygon(tuple([cen[0]+wid,cen[1]]),poly)
        if ((f1 and f2 and f3 and f4) == False):
            return False
    else:
        # print(window)
        # print(headtail)
        # print(cen)
        # print(wid)
        # print(a.get_length())
        ishori = True
        if(a.get_shape() != "Horizontal"):
            ishori = False
        l = a.get_length() / 2
        w = a.get_width()
        if(ishori):
            p1 = [x0 + l + w,y0+w]
            p2 = [x0 + l + w,y1-w]
            p3 = [x1 - l - w,y1-w]
            p4 = [x1 - l - w,y0+w]
        else:
            p1 = [x0+w,y0+l+w]
            p2 = [x0+w,y1-l-w]
            p3 = [x1-w,y1-l-w]
            p4 = [x1-w,y0+l+w]
        poly = [p1,p2,p3,p4]
        # print(ishori)
        # print(poly)
        d1 = point_segment_distance(cen,(p1,p2))
        d2 = point_segment_distance(cen,(p3,p2))
        d3 = point_segment_distance(cen,(p3,p4))
        d4 = point_segment_distance(cen,(p1,p4))

        if(d1 == 0 or d2 == 0 or d3 == 0 or d4 == 0):
            return False
        if(is_point_in_polygon(cen,poly) == False):
            return False
    return True


def is_point_in_polygon(point, polygon):
    """Determine whether a point is in a parallelogram.
    Note: The vertex of the parallelogram should be clockwise or counter-clockwise.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of 4 vertices of the parallelogram.
    """
    x,y = point
    # print(point)
    
    p1 = polygon[0]
    p2 = polygon[1]
    p3 = polygon[2]
    p4 = polygon[3]
    if(x < min(p1[0],p2[0],p3[0],p4[0]) or x > max(p1[0],p2[0],p3[0],p4[0]) or y < min(p1[1],p2[1],p3[1],p4[1]) or y > max(p1[1],p2[1],p3[1],p4[1])):
       return False
    a = (p2[0] - p1[0]) * (y - p1[1]) - (p2[1] - p1[1]) * (x - p1[0])
    b = (p3[0] - p2[0]) * (y - p2[1]) - (p3[1] - p2[1]) * (x - p2[0])
    c = (p4[0] - p3[0]) * (y - p3[1]) - (p4[1] - p3[1]) * (x - p3[0])
    d = (p1[0] - p4[0]) * (y - p4[1]) - (p1[1] - p4[1]) * (x - p4[0])
    if ((a >= 0 and b >= 0 and c >= 0 and d >= 0) or (a <= 0 and b <= 0 and c <= 0 and d <= 0)):
        return True
    return False


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """

    a = alien
    circle = a.is_circle()
    cen = a.get_centroid()

    if(cen == waypoint):
        return does_alien_touch_wall(alien,walls)
    
    wid = a.get_width()
    if(circle):
        for line in walls:
            # print(line)
            l = tuple([(line[0],line[1]),(line[2],line[3])])
            if(segment_distance(tuple([cen,waypoint]),l) <= wid):
                # print(point_segment_distance(cen,l),wid)
                return True
    else:
        ishori = True
        if(a.get_shape() != "Horizontal"):
            ishori = False
        
        for line in walls:
            # print(line)
            l = tuple([(line[0],line[1]),(line[2],line[3])])
            if(ishori):
                s = list(cen)
                e = list(waypoint)
                s[0] -= (a.get_length() / 2)
                e[0] -= (a.get_length() / 2)
                dist1 = segment_distance(tuple([s,e]),l)

                s[0] += (a.get_length())
                e[0] += (a.get_length())
                dist2 = segment_distance(tuple([s,e]),l)
                if(min([dist1,dist2]) <= wid):
                    return True
            else:
                s = list(cen)
                e = list(waypoint)

                s[1] -= (a.get_length() / 2)
                e[1] -= (a.get_length() / 2)
                dist1 = segment_distance(tuple([s,e]),l)

                s[1] += (a.get_length())
                e[1] += (a.get_length())
                dist2 = segment_distance(tuple([s,e]),l)

                if(min([dist1,dist2]) <= wid):
                    return True
    return False


def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    x,y = p
    x1, y1 = s[0]
    x2, y2 = s[1]

    d2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1); ###AB*AP

    if (d2 > 0):
        cross /= d2
    if (cross < 0):
        # retrun ap
        return np.sqrt((x-x1)**2 + (y-y1)**2)
    if (cross > 1):
        # return bp
        return np.sqrt((x-x2)**2 + (y-y2)**2)

    #return pq = AP X AB / |AB|
    a1 = x - x1
    b1 = y - y1
    a2 = x2 - x1
    b2 = y2 - y1
    return abs((a1 * b2 - a2 * b1) / np.sqrt(d2))


def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    x1, y1 = s1[0]
    x2, y2 = s1[1]
    x3, y3 = s2[0]
    x4, y4 = s2[1]
    if(max(x3,x4) < min(x1,x2) or max(y3,y4) < min(y1,y2) or max(x1,x2) < min(x3,x4) or max(y1,y2) < min(y3,y4) ):
        return False
    flag1 = ((x1-x3)*(y4-y3) - (y1-y3)*(x4-x3))* ((x2-x3)*(y4-y3) - (y2-y3)*(x4-x3)) # sign of CA X CD * CA X CB
    flag2 = ((x3-x1)*(y2-y1) - (y3-y1)*(x2-x1))* ((x4-x1)*(y2-y1) - (y4-y1)*(x2-x1)) # sign of AB X AC * AB X AD
    if(flag1 > 0 or flag2 > 0):
        return False
    return True
def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """

    if (do_segments_intersect(s1,s2)):
        return 0
    d1=point_segment_distance(s1[0], tuple([s2[0],s2[1]]))
    d2=point_segment_distance(s1[1], tuple([s2[0],s2[1]]))
    d3=point_segment_distance(s2[0], tuple([s1[0],s1[1]]))
    d4=point_segment_distance(s2[1], tuple([s1[0],s1[1]]))
    return min(d1,d2,d3,d4)



if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
 <font face="Arial" size=2>
<p>Microsoft VBScript runtime </font> <font face="Arial" size=2>error '800a01b6'</font>
<p>
<font face="Arial" size=2>Object doesn't support this property or method: 'Response.Close'</font>
<p>
<font face="Arial" size=2>/fair/getfile.asp</font><font face="Arial" size=2>, line 73</font> 