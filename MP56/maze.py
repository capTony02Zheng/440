# maze.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Joshua Levine (joshua45@illinois.edu) and Jiaqi Gun
"""
This file contains the Maze class, which reads in a maze file and creates
a representation of the maze that is exposed through a simple interface.
"""

import copy
from state import MazeState, euclidean_distance
from geometry import does_alien_path_touch_wall, does_alien_touch_wall
# Euclidean distance between two state tuples, of the form (x,y, shape)

class MazeError(Exception):
    pass


class NoStartError(Exception):
    pass


class NoObjectiveError(Exception):
    pass


class Maze:
    def __init__(self, alien, walls, waypoints, goals, move_cache={}, k=5, use_heuristic=True):
        """Initialize the Maze class, which will be navigated by a crystal alien

        Args:
            alien: (Alien), the alien that will be navigating our map
            walls: (List of tuple), List of endpoints of line segments that comprise the walls in the maze in the format
                        [(startx, starty, endx, endx), ...]
            waypoints: (List of tuple), List of waypoint coordinates in the maze in the format of [(x, y), ...]
            goals: (List of tuple), List of goal coordinates in the maze in the format of [(x, y), ...]
            move_cache: (Dict), caching whether a move is valid in the format of
                        {((start_x, start_y, start_shape), (end_x, end_y, end_shape)): True/False, ...}
            k (int): the number of waypoints to check when getting neighbors
        """
        self.k = k
        self.alien = alien
        self.walls = walls

        self.states_explored = 0
        self.move_cache = move_cache
        self.use_heuristic = use_heuristic

        self.__start = (*alien.get_centroid(), alien.get_shape_idx())
        self.__objective = tuple(goals)

        # Waypoints: the alien must move between waypoints (goal is a special waypoint)
        # Goals are also viewed as a part of waypoints
        self.__waypoints = waypoints + goals
        self.__valid_waypoints = self.filter_valid_waypoints()
        self.__start = MazeState(self.__start, self.get_objectives(), 0, self, self.use_heuristic)

        # self.__dimensions = [len(input_map), len(input_map[0]), len(input_map[0][0])]
        # self.__map = input_map

        if not self.__start:
            # raise SystemExit
            raise NoStartError("Maze has no start")

        if not self.__objective:
            raise NoObjectiveError("Maze has no objectives")

        if not self.__waypoints:
            raise NoObjectiveError("Maze has no waypoints")

    def is_objective(self, waypoint):
        """"
        Returns True if the given position is the location of an objective
        """
        return waypoint in self.__objective

    # Returns the start position as a tuple of (row, col, level)
    def get_start(self):
        assert (isinstance(self.__start, MazeState))
        return self.__start

    def set_start(self, start):
        """
        Sets the start state
        start (MazeState): a new starting state
        return: None
        """
        self.__start = start

    # Returns the dimensions of the maze as a (num_row, num_col, level) tuple
    # def get_dimensions(self):
    #     return self.__dimensions

    # Returns the list of objective positions of the maze, formatted as (x, y, shape) tuples
    def get_objectives(self):
        return copy.deepcopy(self.__objective)

    def get_waypoints(self):
        return self.__waypoints

    def get_valid_waypoints(self):
        return self.__valid_waypoints

    def set_objectives(self, objectives):
        self.__objective = objectives

    # TODO VI d
    def filter_valid_waypoints(self):
        """Filter valid waypoints on each alien shape

            Return:
                A dict with shape index as keys and the list of waypoints coordinates as values
        """
        valid_waypoints = {i: [] for i in range(len(self.alien.get_shapes()))}
        for i in range(len(self.alien.get_shapes())):
            temp = []
            for coor in self.get_waypoints():
                newa = self.create_new_alien(coor[0],coor[1],i)
                if(coor == (127,88) and i == 0):
                    print(does_alien_touch_wall(newa,self.walls))
                if (does_alien_touch_wall(newa,self.walls) == False):
                    temp.append(coor)
            valid_waypoints[i] = temp
        # print(valid_waypoints)
        return valid_waypoints

    # TODO VI d
    def get_nearest_waypoints(self, cur_waypoint, cur_shape):
        """Find the k nearest valid neighbors to the cur_waypoint from a list of 2D points.
            Args:
                cur_waypoint: (x, y) waypoint coordinate
                cur_shape: shape index
            Return:
                the k valid waypoints that are closest to waypoint
        """
        nearest_neighbors = []
        tofind = {}
        valid = self.get_valid_waypoints()
        for loc in valid[cur_shape]:
            if(loc == cur_waypoint):
                continue
            dist = euclidean_distance(tuple([cur_waypoint[0],cur_waypoint[1],cur_shape]),tuple([loc[0],loc[1],cur_shape]))
            if(self.is_valid_move(tuple([cur_waypoint[0],cur_waypoint[1],cur_shape]),tuple([loc[0],loc[1],cur_shape]))):
                tofind[tuple([loc,cur_shape])] = dist
        tofindsorted = sorted(tofind.items(), key=lambda x:x[1])
        for i in range(min(len(tofindsorted),self.k)):
            nearest_neighbors.append(tofindsorted[i][0][0])
        # if(cur_waypoint == (113,120) and cur_shape == 2):
        #     print("!DADSD")
        #     print([nearest_neighbors[0][0],nearest_neighbors[0][1],cur_shape])
        #     print(self.is_valid_move(tuple([cur_waypoint[0],cur_waypoint[1],cur_shape]),tuple([nearest_neighbors[0][0],nearest_neighbors[0][1],cur_shape])))
        #     print(does_alien_path_touch_wall(self.create_new_alien(177,114,2),self.walls,[nearest_neighbors[0][0],nearest_neighbors[0][1]]))
        return nearest_neighbors

    def create_new_alien(self, x, y, shape_idx):
        alien = copy.deepcopy(self.alien)
        alien.set_alien_config([x, y, self.alien.get_shapes()[shape_idx]])
        return alien

    # TODO VI d
    def is_valid_move(self, start, end):
        """Check if the position of the waypoint can be reached by a straight-line path from the current position
            Args:
                start: (start_x, start_y, start_shape_idx)
                end: (end_x, end_y, end_shape_idx)
            Return:
                True if the move is valid, False otherwise
        """

        # shapes = self.alien.get_shape_idx(start[2])
        # shapee = self.alien.get_shape_idx(end[2])
        # print("wtff",start,end)
        if(end[2] == 3 or end[2] == -1): return False
        s = self.create_new_alien(start[0],start[1],start[2])
        e = self.create_new_alien(end[0],end[1],end[2])
        
        # if(start[0] == 113 and start[1]==120 and end[0]==177 and end[1]==114):
        #     print()
        #     print("FUCKED!!")
        #     print(start[2],end[2])
        #     print(does_alien_path_touch_wall(s,self.walls,tuple([end[0],end[1]])))

        # print(s.get_centroid(),s.get_shape())
        # print(e.get_centroid(),e.get_shape())
        # print(does_alien_touch_wall(s,self.walls),does_alien_touch_wall(e,self.walls))
        if(does_alien_touch_wall(s,self.walls) == True or does_alien_touch_wall(e,self.walls) == True):
            # print("this fucked")
            return False
        if(does_alien_path_touch_wall(s,self.walls,tuple([end[0],end[1]])) == True):
            return False
        
        return True

    def get_neighbors(self, x, y, shape_idx):
        """Returns list of neighboring squares that can be moved to from the given coordinate
            Args:
                x: query x coordinate
                y: query y coordinate
                shape_idx: query shape index
            Return:
                list of possible neighbor positions, formatted as (x, y, shape) tuples.
        """
        self.states_explored += 1

        nearest = self.get_nearest_waypoints((x, y), shape_idx)
        # print(nearest)
        neighbors = [(*end, shape_idx) for end in nearest]
        # print(x,y,shape_idx, "and neighbor:", neighbors)
        for end in [(x, y, shape_idx - 1), (x, y, shape_idx + 1)]:
            start = (x, y, shape_idx)
            # if(start[0] == 113 and start[1]==120 and end[0]==177 and end[1]==114):
            #     print("FUCKED!!!!!")
            if self.is_valid_move(start, end):
                neighbors.append(end)
        # print(x,y,shape_idx, "and neighbor with self:", neighbors)
        return neighbors
