
import heapq
import random

import numpy as np
from PIL import Image
import os 


def coord_toImg(x,y):
        x = round(x,1)
        y = round(y,1)
        x_translation = 5
        y_translation = 12
        x_scale = 2
        y_scale = 2
         

        x_transformed = int(x * x_scale + x_translation)
        y_transformed = int(y * y_scale + y_translation)

        return x_transformed,y_transformed


def astar_planner(grid, start, goal):
    # Define the possible movements (up, down, left, right)
    movements = [(0, 1), (0, -1), (1, 0), (-1, 0),(1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    # Get the dimensions of the grid
    rows = len(grid)
    cols = len(grid[0])
    
    # Initialize the cost and heuristic dictionaries
    cost = {}
    heuristic = {}
    
    # Initialize the priority queue
    queue = []
    
    # Set the cost of the start node to 0
    cost[start] = 0
    
    # Calculate the heuristic value for the start node
    heuristic[start] = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    
    # Push the start node into the priority queue
    heapq.heappush(queue, (cost[start] + heuristic[start], start))
    
    # Initialize the parent dictionary
    parent = {}
    
    while queue:
        # Pop the node with the lowest cost from the priority queue
        current_cost, current_node = heapq.heappop(queue)
        
        # Check if the current node is the goal node
        if current_node == goal:
            break
        
        # Explore the neighbors of the current node
        for movement in movements:
            neighbor = (current_node[0] + movement[0], current_node[1] + movement[1])
            
            # Check if the neighbor is within the grid boundaries
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # Calculate the cost of moving to the neighbor
                new_cost = cost[current_node] + 1
                
                # Check if the neighbor is a free space and the new cost is lower than the current cost
                if (grid[neighbor[0]][neighbor[1]] > 0) and (neighbor not in cost or new_cost < cost[neighbor]):
                    # Update the cost and heuristic values for the neighbor
                    cost[neighbor] = new_cost
                    heuristic[neighbor] = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    
                    # Push the neighbor into the priority queue
                    heapq.heappush(queue, (cost[neighbor] + heuristic[neighbor], neighbor))
                    
                    # Set the parent of the neighbor to the current node
                    parent[neighbor] = current_node
    
    # Reconstruct the path from the goal node to the start node
    path = []
    current_node = goal
    
    while current_node != start:
        path.append(current_node)
        current_node = parent[current_node]
    
    path.append(start)
    path.reverse()
    
    return path

def get_part_coordinates(corners):
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    
    min_x, max_x = sorted([x1, x2])
    min_y, max_y = sorted([y1, y2])
    
    return [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]








def get_random_paths(elevator_cases,a,clean_binary_array):
    paths = {}
    
    humans = elevator_cases[a]
    for i in humans.keys():    
        if humans[i] == []:
            return 0
        
        if humans[i]['destination'] == None:
            paths[i] = astar_planner(clean_binary_array,humans[i]['origin'],humans[i]['origin'])
        
        else:
             paths[i] = astar_planner(clean_binary_array,humans[i]['origin'],humans[i]['destination'])

    return paths


def get_hum_cases():

    _in1 = [[9.5, 0],[8,-0.5]]
    _in2 = [[9.5, 0.5],[8,0]]

    _door1 =  [[8,0],[6.5,-1]]
    _door2 =  [[8, 1],[6.5,0]]

    _out1 =  [[6 , 5.5],[4 , 2]]
    _out2 =  [[6 , -2],[4 , -5.5]]

    _dest1 =  [[4 , 5.5],[-2 , 2]]
    _dest2=  [[4 , -2],[-2 , -5.5]]

    in1 = [coord_toImg(_in1[0][0],_in1[0][1]),
                                coord_toImg(_in1[1][0],_in1[1][1])]    

    in2 = [coord_toImg(_in2[0][0],_in2[0][1]),
                                coord_toImg(_in2[1][0],_in2[1][1])]    

    door1 = [coord_toImg(_door1[0][0],_door1[0][1]),
                                coord_toImg(_door1[1][0],_door1[1][1])]   

    door2 = [coord_toImg(_door2[0][0],_door2[0][1]),
                                coord_toImg(_door2[1][0],_door2[1][1])]   

    out1 = [coord_toImg(_out1[0][0],_out1[0][1]),
                                coord_toImg(_out1[1][0],_out1[1][1])] 

    out2 = [coord_toImg(_out2[0][0],_out2[0][1]),
                                coord_toImg(_out2[1][0],_out2[1][1])] 


    dest1 = [coord_toImg(_dest1[0][0],_dest1[0][1]),
                                coord_toImg(_dest1[1][0],_dest1[1][1])] 


    dest2 = [coord_toImg(_dest2[0][0],_dest2[0][1]),
                                coord_toImg(_dest2[1][0],_dest2[1][1])] 

    _in = [in1 ,in2]
    _door = [door1,door2]
    _out = [out1,out2]
    _dest = [dest1,dest2]


    elevator_cases = {
                    2: {1:{'origin':random.choice(get_part_coordinates(random.choice(_door))),'destination':random.choice(get_part_coordinates(random.choice(_dest)))}}, 
                    3:{1:{'origin':random.choice(get_part_coordinates(random.choice(_in))),'destination':None}},
                    4:{1:{'origin':random.choice(get_part_coordinates(random.choice(_door))),'destination':random.choice(get_part_coordinates(random.choice(_dest)))},
                        2:{'origin':random.choice(get_part_coordinates(random.choice(_in))),'destination':None}},
                    5:{1:{'origin':random.choice(get_part_coordinates(random.choice(_out))),'destination':random.choice(get_part_coordinates(random.choice(_in)))}},
                    6:{1:{'origin':random.choice(get_part_coordinates(_door[0])),'destination':random.choice(get_part_coordinates(_dest[0]))},
                        2:{'origin':random.choice(get_part_coordinates(_door[1])),'destination':random.choice(get_part_coordinates(_dest[1]))}},
                    7:{1:{'origin':random.choice(get_part_coordinates(_in[0])),'destination': None},
                        2:{'origin':random.choice(get_part_coordinates(_in[1])),'destination':None}},
                    8:{1:{'origin':random.choice(get_part_coordinates(_in[0])),'destination':None},
                        2:{'origin':random.choice(get_part_coordinates(random.choice(_door))),'destination':random.choice(get_part_coordinates(random.choice(_dest)))},
                        3:{'origin':random.choice(get_part_coordinates(random.choice(_out))),'destination':random.choice(get_part_coordinates(_in[1]))}},
                    9:{1:{'origin':random.choice(get_part_coordinates(_out[0])),'destination':random.choice(get_part_coordinates(_in[1]))},
                        2:{'origin':random.choice(get_part_coordinates(_out[1])),'destination':random.choice(get_part_coordinates(_in[0]))}},
                    10:{1:{'origin':random.choice(get_part_coordinates(random.choice(_out))),'destination':random.choice(get_part_coordinates(random.choice(_in)))},
                        2:{'origin':random.choice(get_part_coordinates(random.choice(_door))),'destination':random.choice(get_part_coordinates(random.choice(_dest)))}},
                    11:{1:{'origin':random.choice(get_part_coordinates(random.choice(_out))),'destination':random.choice(get_part_coordinates(_in[1]))},
                        2:{'origin':random.choice(get_part_coordinates(_in[0])),'destination':None}},
                    12:{1:{'origin':random.choice(get_part_coordinates(_door[0])),'destination':random.choice(get_part_coordinates(_dest[0]))},
                        2:{'origin':random.choice(get_part_coordinates(_door[1])),'destination':random.choice(get_part_coordinates(_dest[1]))},
                        3: {'origin':random.choice(get_part_coordinates(random.choice(_out))),'destination':random.choice(get_part_coordinates(random.choice(_in)))}}
    }
    return elevator_cases
