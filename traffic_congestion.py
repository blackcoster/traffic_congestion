import momepy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from iduedu import get_adj_matrix_gdf_to_gdf
import matplotlib.pyplot as plt
import contextily as ctx

def get_OD_mx(blocks, nodes, walk_graph, drive_graph, crs):

    blocks.to_crs(crs,inplace=True)
    blocks.loc[blocks['density'].isna(), 'density'] = 0.0
    blocks.loc[blocks['diversity'].isna(), 'diversity'] = 0.0
    scaler = MinMaxScaler()
    blocks[["density", "diversity"]] = scaler.fit_transform(blocks[["density", "diversity"]])
    blocks = blocks.set_geometry('geometry')
    landuse_coeff = {
        None: 0.06,
        'industrial': 0.25,
        'business': 0.3,
        'special': 0.1,
        'transport': 0.1,
        'residential': 0.1,
        'agriculture': 0.05,
        'recreation': 0.05
    }
    blocks['lu_coeff'] = blocks['land_use'].apply(lambda x: landuse_coeff.get(x, 0))
    blocks['attractiveness'] = blocks['density']+blocks['diversity']+blocks['lu_coeff']
    blocks[['population', 'attractiveness']] = scaler.fit_transform(blocks[['population', 'attractiveness']])
    
    nodes.to_crs(crs,inplace=True)
    
    walk_mx = get_adj_matrix_gdf_to_gdf(
            blocks,
            nodes,
            walk_graph,
            weight="time_min",
            dtype=np.float64,
        )
    
    walk_dict = {}

    for i, row in walk_mx.iterrows():
        walk_dict[i] = []
        for j, value in row.items():  # Iterate over columns in the row
            if value <= 10:  # Check condition
                walk_dict[i].append((j,value))
        if len(walk_dict[i])==0:
            walk_dict[i].append((row.idxmin(),row.min()))


    # Исходный словарь {блок: [(остановка, расстояние)]}
    block_to_stops = walk_dict.copy()

    # Новый словарь {остановка: [(блок, коэффициент)]}
    block_to_weights = {}

    for block1, stops1 in block_to_stops.items():
        # Разбираем список остановок
        stop_ids = np.array([stop[0] for stop in stops1])  # Индексы остановок
        distances = np.array([stop[1] if stop[1] > 0 else 0.1 for stop in stops1], dtype=np.float64)  # Расстояния

        # Вычисляем обратные веса
        weights = 1 / distances  # Чем ближе, тем больше вес

        # Нормируем веса, чтобы сумма по блоку была 1
        weights_normalized = weights / weights.sum()

        # Записываем результат
        block_to_weights[block1] = list(zip(stop_ids, weights_normalized))

    stops_dict = {s:{'att': 0, 'pop': 0} for s in list(nodes.index) }

    for key,value in block_to_weights.items():
        for stahp,k in value:

            stops_dict[stahp]['att'] += blocks.iloc[key]['attractiveness'] * k
            stops_dict[stahp]['pop'] += blocks.iloc[key]['population'] * k

    nodes['att'] = [v['att'] for k,v in stops_dict.items()]
    nodes['pop'] = [v['pop'] for k,v in stops_dict.items()]

    mx_stopstop = get_adj_matrix_gdf_to_gdf(nodes,nodes,drive_graph,'length_meter',dtype=np.float64)
    # Ensure no division by zero
    adj_mx = mx_stopstop.replace(0, np.nan)

    # Compute the OD matrix using the Gravity Model formula
    od_matrix = pd.DataFrame(
        np.outer(nodes["pop"], nodes["att"]) / adj_mx,
        index=adj_mx.index,
        columns=adj_mx.columns
    )

    # Fill NaN values (from division by zero) with 0
    od_matrix = od_matrix.fillna(0)

    return od_matrix


def get_road_congestion(OD_mx,graph):
    import networkx as nx

    path = dict(nx.all_pairs_dijkstra_path(graph,weight='time_min'))

    for u, v, d in graph.edges(data=True):
        d['congestion'] = 0.0
        
    for i in range(len(n)):
        for j in range(len(n)):
            if  i  in path and j in path[i]:
                p = path[i][j]
                for k in range(len(p)-1):
                    graph[p[k]][p[k+1]][0]['congestion'] += OD_mx[i][j]
    return graph




def visualize_congestion(graph,scale_factor=250,label='traffic congestion'):
    n2,e2 = momepy.nx_to_gdf(graph)  
    # Convert your edges GeoDataFrame to Web Mercator projection (EPSG:3857)
    edges_proj = e2.to_crs(epsg=3857).copy()

    # Set a scale factor for the edge line widths
    scale_factor = 250 # Adjust as needed

    fig, ax = plt.subplots(figsize=(8, 8))

    # Set axis limits with a 10% margin around the total bounds of your edges
    xmin, ymin, xmax, ymax = edges_proj.total_bounds
    x_margin = (xmax - xmin) * 0.1 / 2
    y_margin = (ymax - ymin) * 0.1 / 2
    ax.set_xlim(xmin - x_margin, xmax + x_margin)
    ax.set_ylim(ymin - y_margin, ymax + y_margin)

    # Add a basemap using contextily (less textured for clarity)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=edges_proj.crs, alpha=0.7)

    # Plot the edges with line widths scaled by the 'business' attribute
    edges_proj.plot(ax=ax, color="blue", 
                    linewidth=edges_proj['congestion'] * scale_factor, 
                    alpha=0.8)

    plt.title(label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    plt.show()
