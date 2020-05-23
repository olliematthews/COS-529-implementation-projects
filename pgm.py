import pickle
import numpy as np
import argparse
import networkx as nx


def load_graph(filename):
    'load the graphical model (DO NOT MODIFY)'
    return pickle.load(open(filename, 'rb'))


def inference_brute_force(G):
    '''
    Perform probabilistic inference on graph G, and compute the gradients of the log-likelihood
    Input: 
        G: 
            a Graph object in the NetworkX library (version 2.2, https://networkx.github.io/)
        K: 
            G.graph['K']
        unary potentials: 
            G.nodes[v]['unary_potential'], a 1-d numpy array of length K
        binary potentials: 
            G.edges[u, v]['binary_potential'], a 2-d numpy array of shape K x K
        assignment for computing the gradients: 
            G.nodes[v]['assignment'], an integer within [0, K - 1]
    Output:
        G.nodes[v]['marginal_prob']: 
            the marginal probability distribution for v, a 1-d numpy array of length K
        G.graph['v_map']: 
            the MAP assignment, a 1-d numpy arrary of length n, where n is the number of vertices
        G.nodes[v]['gradient_unary_potential']: 
            the gradient of the log-likelihood w.r.t. the unary potential of vetext v, a 1-d numpy array of length K
        G.edges[u, v]['gradient_binary_potential']: 
            the gradient of the log-likelihood w.r.t. the binary potential of edge (u, v), a 2-d numpy array of shape K x K
    '''
    # initialize the output buffers
    for v in G.nodes:
        G.nodes[v]['marginal_prob'] = np.zeros(G.graph['K'])
        G.nodes[v]['gradient_unary_potential'] = np.zeros(G.graph['K'])
    for e in G.edges:
        G.edges[e]['gradient_binary_potential'] = np.zeros((G.graph['K'], G.graph['K']))
    G.graph['v_map'] = np.zeros(len(G.nodes))
    
    # Create a list for the unary potentials and assignment for easier access
    potentials = []
    assignment = []
    for i in G.nodes:
        potentials.append(G.nodes[i]['unary_potential'])
        assignment.append(G.nodes[i]['assignment'])
    assignment = np.array(assignment)
    
    # Initialise Z, and a running counter for the maximum potential encountered
    # for finding the MAP estimate
    Z = 0
    p_max = 0
    
    # Loop through every possible combination of values
    for i in range(G.graph['K'] ** len(G.nodes)):
        # Find the y values for the ith combination
        us = [(i // G.graph['K'] ** j) % G.graph['K'] for j in range(len(G.nodes))]
        
        # Sum up potentials of nodes
        prob = np.prod([pot[u] for pot, u in zip(potentials, us)])
        
        # Sum up the potentials of edges
        for edge in G.edges:
            prob *= G.edges[edge]['binary_potential'][us[edge[0]],us[edge[1]]]
        
        # Update Z, and the relevant marginal probabilities and log gradients
        # for the Z term
        Z += prob
        for i, u in enumerate(us):
            G.nodes[i]['marginal_prob'][u] += prob
            G.nodes[i]['gradient_unary_potential'][u] -= prob / potentials[i][u]
        for edge in G.edges:
            G.edges[edge]['gradient_binary_potential'][us[edge[0]],us[edge[1]]] -= prob / G.edges[edge]['binary_potential'][us[edge[0]],us[edge[1]]]
            
        # Keep running track of MAP
        if prob > p_max:
            p_max = prob
            G.graph['v_map'] = np.array(us)
            
    # Normalise the probabilities
    for i in G.nodes:
        G.nodes[i]['marginal_prob'] /= Z
        G.nodes[i]['gradient_unary_potential'] /= Z
    for edge in G.edges:
        G.edges[edge]['gradient_binary_potential'] /= Z
        
    # Complete the rest of the gradient terms which depend on the assignments
    for i, u in enumerate(assignment):
        G.nodes[i]['gradient_unary_potential'][u] += 1 / G.nodes[i]['unary_potential'][u]
    for edge in G.edges:
        u = assignment[edge[0]]
        v = assignment[edge[1]]
        G.edges[edge]['gradient_binary_potential'][u,v] += 1 / G.edges[edge]['binary_potential'][u,v]


def inference(G):
    '''
    Perform probabilistic inference on graph G, and compute the gradients of the log-likelihood
    Input: 
        G: 
            a Graph object in the NetworkX library (version 2.2, https://networkx.github.io/)
        K: 
            G.graph['K']
        unary potentials: 
            G.nodes[v]['unary_potential'], a 1-d numpy array of length K
        binary potentials: 
            G.edges[u, v]['binary_potential'], a 2-d numpy array of shape K x K
        assignment for computing the gradients: 
            G.nodes[v]['assignment'], an integer within [0, K - 1]
    Output:
        G.nodes[v]['marginal_prob']: 
            the marginal probability distribution for v, a 1-d numpy array of length K
        G.graph['v_map']: 
            the MAP assignment, a 1-d numpy arrary of length n, where n is the number of vertices
        G.nodes[v]['gradient_unary_potential']: 
            the gradient of the log-likelihood w.r.t. the unary potential of vetext v, a 1-d numpy array of length K
        G.edges[u, v]['gradient_binary_potential']: 
            the gradient of the log-likelihood w.r.t. the binary potential of edge (u, v), a 2-d numpy array of shape K x K
    '''
    # initialize the output buffers. All edges also get a message in each
    # direction, and an map value and index for the upwards direction
    for v in G.nodes:
        G.nodes[v]['marginal_prob'] = np.zeros(G.graph['K'])
        G.nodes[v]['gradient_unary_potential'] = np.zeros(G.graph['K'])
    for e in G.edges:
        G.edges[e]['gradient_binary_potential'] = np.zeros((G.graph['K'], G.graph['K']))
        G.edges[e]['message_up'] = np.zeros([G.graph['K'],])
        G.edges[e]['message_down'] = np.zeros([G.graph['K'],])
        G.edges[e]['map_up'] = np.zeros([G.graph['K'],])
        G.edges[e]['map_up_index'] = np.zeros([G.graph['K'],]).astype(int)
    G.graph['v_map'] = np.zeros(len(G.nodes)).astype(int)
            
            
    # Take lowest number as root, and all other nodes with one edge as leaves
    leaves = [node for node in G.nodes if len(G.edges(node)) == 1]
    root = leaves[0]
    leaves = leaves[1:]
    
    # Upwards pass
    G = belief_propogation_up(G, leaves.copy(), root)
    
    # Downwards pass
    G = belief_propogation_down(G, [root], leaves)
    
    # Set marginals and gradients
    G = set_marginals_gradients(G)
    
    # Set map
    G = set_map(G, root)
    
def belief_propogation_up(G, node_list, root):
    '''
    Does an upwards pass of the tree, updating each edge with an upwards 
    message and map values
    Input: 
        G: 
            a Graph object in the NetworkX library (version 2.2, https://networkx.github.io/)
        node_list: 
            a list of nodes which need to be explored
        root:
            the root node of the graph
    Output:
        G.edges[e]['message_up']: 
            the message up over an edge
        G.edges[e]['map_up']: 
            the map value of the node below given a value of the node above
        G.edges[e]['map_up_index']: 
            the index for the map value
            
    '''
    
    # Base case - return if there are no more nodes to investigate. Else take
    # nodes off the list in a FIFO manner
    if node_list == []:
        return G
    node = node_list.pop(0)
    
    # Target edges are edges where message up has not been evaluated yet. Done
    # edges have already been evaluated
    target_edges = []
    done_edges = []
    for edge in G.edges(node):
        if np.sum(G.edges[edge]['message_up']) == 0:
            target_edges.append(edge)
        else:
            done_edges.append(edge)

    # Only nodes where all the surrounding edges bar one have been evaluated 
    # can be evaluated
    if len(target_edges) == 1:       
        for k in range(G.graph['K']):
            # Update the message, map, and map index for each y value on the 
            # given edge
            boi = G.edges[target_edges[0]]['binary_potential'][k,:] * G.nodes[node]['unary_potential']
            max_boi = boi * np.prod([G.edges[e]['map_up'] for e in done_edges],axis = 0)
            boi *= np.prod([G.edges[e]['message_up'] for e in done_edges], axis = 0)

            G.edges[target_edges[0]]['message_up'][k] = sum(boi)
            G.edges[target_edges[0]]['map_up_index'][k] = np.argmax(max_boi)
            G.edges[target_edges[0]]['map_up'][k] = max_boi[G.edges[target_edges[0]]['map_up_index'][k]]
        # Normalise for numerical stability
        G.edges[target_edges[0]]['message_up'] /= np.sum(G.edges[target_edges[0]]['message_up'])
        G.edges[target_edges[0]]['map_up'] /= np.sum(G.edges[target_edges[0]]['map_up'])
        # Move onto next node up, provided it is not the root
        next_node = [n for n in target_edges[0] if not n == node][0]
        if not next_node in node_list and not next_node == root:
            node_list.append(next_node)
    # If the node cannot be evaluated yet, move onto the next node in the list   
    else:
        node_list.append(node)
        
    return belief_propogation_up(G, node_list, root)

def belief_propogation_down(G, node_list, leaves):
    '''
    Does a downwards pass of the tree, updating each edge with a downwards 
    message
    Input: 
        G: 
            a Graph object in the NetworkX library (version 2.2, https://networkx.github.io/)
        node_list: 
            a list of nodes which need to be explored
        root:
            the root node of the graph
    Output:
        G.edges[e]['message_down']: 
            the message down over an edge
    '''
    # Base case - return if there are no more nodes to investigate. Else take
    # nodes off the list in a FIFO manner
    if node_list == []:
        return G
    node = node_list.pop(0)
    
    # Keep track of the edges going up and down towards the node. There should 
    # only be one going down towards the node.
    down_edges = []
    up_edges = []
    for edge in G.edges(node):
        if 2 * node > sum(edge):
            down_edges.append(edge)
        else:
            up_edges.append(edge)
                
    # Iterate through the edges going up to the node, evaluating one at a time
    for target_edge in up_edges:
        # All other edges are done since we have done a upwards pass already
        up_edges_done = [edge for edge in up_edges if not edge == target_edge]
        for k in range(G.graph['K']):
            # Evaluate the downwards message for each y value of the node
            boi = G.edges[target_edge]['binary_potential'][:,k] * G.nodes[node]['unary_potential'] 
            boi *= (np.prod([G.edges[e]['message_down'] for e in down_edges], axis = 0)) * np.prod([G.edges[e]['message_up'] for e in up_edges_done], axis = 0)
            G.edges[target_edge]['message_down'][k] = sum(boi)

        # Normalise for numerical stability
        G.edges[target_edge]['message_down'] /= np.sum(G.edges[target_edge]['message_down'])

        # Add the next nodes down to the list provided they are not leaves
        next_node = [n for n in target_edge if not n == node][0]
        if not next_node in node_list and not next_node in leaves:
            node_list.append(next_node)
        
    return belief_propogation_down(G, node_list, leaves)

def set_marginals_gradients(G):
    '''
    Sets the marginals and gradients using the calculated messages
    Input: 
        G: 
            a Graph object in the NetworkX library (version 2.2, https://networkx.github.io/)
    Output:
        G.nodes[v]['marginal_prob']: 
            the marginal probability distribution for v, a 1-d numpy array of length K
        G.nodes[v]['gradient_unary_potential']: 
            the gradient of the log-likelihood w.r.t. the unary potential of vetext v, a 1-d numpy array of length K
        G.edges[u, v]['gradient_binary_potential']: 
            the gradient of the log-likelihood w.r.t. the binary potential of edge (u, v), a 2-d numpy array of shape K x K
    '''
    for node in G.nodes:
        # Get the upwards and downwards edges
        down_edges = []
        up_edges = []
        for edge in G.edges(node):
            if 2 * node > sum(edge):
                down_edges.append(edge)
            else:
                up_edges.append(edge)
        
        # Update the marginals
        G.nodes[node]['marginal_prob'] = np.prod([G.edges[e]['message_down'] for e in down_edges], axis = 0)
        G.nodes[node]['marginal_prob'] *= np.prod([G.edges[e]['message_up'] for e in up_edges], axis = 0)
        G.nodes[node]['marginal_prob'] *= G.nodes[node]['unary_potential']
        G.nodes[node]['marginal_prob'] /= np.sum(G.nodes[node]['marginal_prob'])
        # Update the gradients
        G.nodes[node]['gradient_unary_potential'] = - G.nodes[node]['marginal_prob'] / G.nodes[node]['unary_potential']
        G.nodes[node]['gradient_unary_potential'][G.nodes[node]['assignment']] += 1 / G.nodes[node]['unary_potential'][G.nodes[node]['assignment']]
        
    for edge in G.edges: 
        # Get the edge in ascending order to tell which is the top node
        edge_sorted = list(edge).copy()
        edge_sorted.sort()
        node_top = edge_sorted[0]
        node_bottom = edge_sorted[1]
        
        # Get the upwards and downwards edges for the top node and the
        # downwards edges for the bottom node
        down_edges_top = []
        up_edges_top = []
        up_edges_bottom = []
        for e in G.edges(node_top):
            if 2 * node_top > sum(e):
                down_edges_top.append(e)
            elif not e == edge:
                up_edges_top.append(e)
                
        for e in G.edges(node_bottom):
            if 2 * node_bottom < sum(e):
                up_edges_bottom.append(e)
        
        # Evaluate the combined probabilities for the edge
        binary_prob = G.edges[edge]['binary_potential'].copy()
        binary_prob *= G.nodes[node_top]['unary_potential'][:,None]
        binary_prob *= G.nodes[node_bottom]['unary_potential'][None,:]
        if not len(down_edges_top) == 0:
            binary_prob *= np.prod([G.edges[e]['message_down'] for e in down_edges_top], axis = 0)[:,None]
        if not len(up_edges_top) == 0:
            binary_prob *= np.prod([G.edges[e]['message_up'] for e in up_edges_top], axis = 0)[:,None]
        if not len(up_edges_bottom) == 0:
            binary_prob *= np.prod([G.edges[e]['message_up'] for e in up_edges_bottom], axis = 0)[None,:]
        binary_prob /= np.sum(binary_prob)
         
        # Evaluate the gradients for the edge
        G.edges[edge]['gradient_binary_potential'] = - binary_prob/ G.edges[edge]['binary_potential']
        G.edges[edge]['gradient_binary_potential'][tuple([G.nodes[n]['assignment'] for n in edge_sorted])] += 1 / G.edges[edge]['binary_potential'][tuple([G.nodes[n]['assignment'] for n in edge_sorted])]
    return G

def set_map(G, root):
    '''
    Sets the map using the map messages
    Input: 
        G: 
            a Graph object in the NetworkX library (version 2.2, https://networkx.github.io/)
    Output:
        G.graph['v_map']: 
            the MAP assignment, a 1-d numpy arrary of length n, where n is the number of vertices
    '''
    
    # For the root, all edges are up edges
    up_edges = []
    next_nodes = []
    for edge in G.edges(root):
        up_edges.append(edge)
        next_nodes.append([n for n in edge if not n == root][0])

    # Take the argmax for the root
    maxes = G.nodes[root]['unary_potential']
    maxes *= np.squeeze(np.array([G.edges[e]['map_up'] for e in G.edges(root)]))
    G.graph['v_map'][0] = np.argmax(maxes)
    
    # Go down the tree, updating each node map value depending on the value
    # taken for the higher node
    for edge, node in zip(up_edges, next_nodes):
        G = recursive_root_set(G, node, edge, G.graph['v_map'][0])
    return G

def recursive_root_set(G, node, edge, yi):
    '''
    Sets the map for a node given the map of the node above
    Input: 
        G: 
            a Graph object in the NetworkX library (version 2.2, https://networkx.github.io/)
        node: 
            the node in question
        edge: 
            the edge connecting the node to the previous node
        yi:
            the map value of the previous node
    Output:
        G.graph['v_map'][node]: 
            the MAP assignment for the given node
    '''
    # Get the edges coming upwards towards the node, and thus the next nodes
    up_edges = []
    next_nodes = []
    for e in G.edges(node):
        if 2 * node < sum(e):
            up_edges.append(e)
            next_nodes.append([n for n in e if not n == node][0])


    # Update the map using the map indexes
    G.graph['v_map'][node] = G.edges[edge]['map_up_index'][yi]

    # Call on the next nodes
    for edge, next_node in zip(up_edges, next_nodes):
        G = recursive_root_set(G, next_node, edge, G.graph['v_map'][node])
    return G


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='The input graph')
    args = parser.parse_args()
    G = load_graph(args.input)
    # inference_brute_force(G)
    inference(G)
    pickle.dump(G, open('results_' + args.input, 'wb'))