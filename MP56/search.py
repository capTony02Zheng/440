# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq


# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze


def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI
def astar(maze):
    visited_states = {maze.get_start(): (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    # print("        swadasd ",maze.get_start())
    heapq.heappush(frontier, maze.get_start())
    
    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    # Your code here ---------------
    found = False
    while (len(frontier) > 0):
        # print(len(frontier))
        # print(len(visited_states))
        top = heapq.heappop(frontier)
        # print(top,visited_states.get(top))
        if (top.is_goal()):
            found = True
            break
        neighbors = top.get_neighbors()

        for nei in neighbors:
            if (nei in visited_states.keys()):
                #already visited
                if (visited_states[nei][1] > nei.dist_from_start):
                    # print("tod dis from s:" ,top.dist_from_start)
                    visited_states[nei] = (top,nei.dist_from_start)
                    heapq.heappush(frontier, nei)

            else:
                #not visited yet
                visited_states[nei] = (top,nei.dist_from_start)
                heapq.heappush(frontier, nei)
        # print("one rim")
        # print(" ")
        
        
        
    # ------------------------------
    if (found == True):
        return backtrack(visited_states, top)
    # if you do not find the goal return an empty list
    return None


# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
# TODO: VI
def backtrack(visited_states, current_state):
    path = []
    # Your code here ---------------

    trackstate = current_state

    while (visited_states[trackstate][0] is not None):
        path.append(trackstate)
        trackstate = visited_states[trackstate][0]

    path.append(trackstate)
    # ------------------------------
    # print(path)
    return path[::-1]
    return None

#python3 mp5_6.py --map Test1 --config maps/test_config.txt --save-maze shit1.txt 
