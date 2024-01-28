from shapely import geometry, ops
from shapely.geometry import Point, mapping, LineString, MultiPoint
import numpy as np
import rasterio
from rasterio.plot import show
import networkx as nx
import geopandas as gpd
import os
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

from cartopy import crs
from scipy.ndimage import filters
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm


def create_graph_and_paths(network_links, points):
    network_links['dpn_length_normalised'] = network_links['dpn'] * network_links['length']

    graph = nx.DiGraph()
    for index, row in network_links.iterrows():
        graph.add_edge(row['startnode'], row['endnode'], fid=row['fid'], length=row.length, dpn=row.dpn_length_normalised,
                       angle=row.angle, surface_cost=row.surface_cost, total_time=row.total_time)

    weighted_path_dpn = nx.dijkstra_path(graph, source=points[0], target=points[1], weight='dpn')
    weighted_path_surface_cost = nx.dijkstra_path(graph, source=points[0], target=points[1], weight='surface_cost')

    return weighted_path_dpn, weighted_path_surface_cost, graph


def rank_exponent_formula(list,exponent):
    weight = []
    for i in list:
        weight.append((len(list)-i+1)**exponent)

    total = sum(weight)

    weight_normalised = []
    for i in weight:
        weight_normalised.append(i/total)

    return weight_normalised


def create_rank_exponent(exponent):

    easy = [3,4,1,2]
    int = [4,2,1,3]
    chal = [4,1,3,2]

    easy_normalised = rank_exponent_formula(easy,exponent)
    intermediate_normalised = rank_exponent_formula(int,exponent)
    challenging_normalised = rank_exponent_formula(chal,exponent)

    return easy_normalised, intermediate_normalised, challenging_normalised


def create_weighted_graph_and_paths(network_links, points):

    # normalise to length of the links
    network_links['angle_length_normalised'] = network_links['angle'] * network_links['length']
    network_links['dpn_length_normalised'] = network_links['dpn'] * network_links['length']
    network_links['surface_cost_length_normalised'] = network_links['surface_cost'] * network_links['length']

    # standard normalisation
    network_links['angle_normalised'] = network_links['angle_length_normalised'].div(network_links['angle_length_normalised'].sum())
    network_links['total_time_normalised'] = network_links['total_time'].div(network_links['total_time'].sum())
    network_links['dpn_normalised'] = network_links['dpn_length_normalised'].div(network_links['dpn_length_normalised'].sum())
    network_links['surface_cost_normalised'] = network_links['surface_cost_length_normalised'].div(network_links['surface_cost_length_normalised'].sum())

    easy, intermediate, challenging = create_rank_exponent(3)

    graph = nx.DiGraph()
    for index, row in network_links.iterrows():
        # easy
        angle = (row.angle_normalised * easy[0]) * 100000
        travel_time = (row.total_time_normalised * easy[1]) * 100000
        dpn = (row.dpn_normalised * easy[2]) * 100000
        surface_cost = (row.surface_cost_normalised * easy[3]) * 100000
        # Weighted Sum method
        wsm_easy = angle + travel_time + dpn + surface_cost

        # intermediate
        angle = (row.angle_normalised * intermediate[0]) * 100000
        travel_time = (row.total_time_normalised * intermediate[1]) * 100000
        dpn = (row.dpn_normalised * intermediate[2]) * 100000
        surface_cost = (row.surface_cost_normalised * intermediate[3]) * 100000
        # Weighted Sum method
        wsm_intermediate = angle + travel_time + dpn + surface_cost

        # intermediate
        angle = (row.angle_normalised * challenging[0]) * 100000
        travel_time = (row.total_time_normalised * challenging[1]) * 100000
        dpn = (row.dpn_normalised * challenging[2]) * 100000
        surface_cost = (row.surface_cost_normalised * challenging[3]) * 100000
        # Weighted Sum method
        wsm_difficult = angle + travel_time + dpn + surface_cost

        graph.add_edge(row['startnode'], row['endnode'], fid=row['fid'], length=row.length,
                       wsm_easy=wsm_easy, wsm_intermediate=wsm_intermediate,
                       wsm_difficult=wsm_difficult)

    weighted_path_easy = nx.dijkstra_path(graph, source=points[0], target=points[1], weight='wsm_easy')
    weighted_path_intermediate = nx.dijkstra_path(graph, source=points[0], target=points[1],
                                                  weight='wsm_intermediate')
    weighted_path_difficult = nx.dijkstra_path(graph, source=points[0], target=points[1], weight='wsm_difficult')

    return weighted_path_easy, weighted_path_intermediate, weighted_path_difficult,graph


def create_path_gpd(weighted_path, network_links, graph):
    geom = []
    links = []
    length = []
    dpn = []
    angle = []
    surface_cost = []
    total_time = []
    wsm_easy = []
    wsm_intermediate = []
    first_node = weighted_path[0]
    for node in weighted_path[1:]:
        link_fid = graph.edges[first_node, node]['fid']
        links.append(link_fid)
        row = network_links.loc[network_links['fid'] == link_fid]
        geom.append(row['geometry'].cascaded_union)
        length.append(row.length.values[0])
        dpn.append(row.dpn.values[0])
        angle.append(row.angle.values[0])
        surface_cost.append(row.surface_cost.values[0])
        total_time.append(row.total_time.values[0])
        wsm_easy.append(row.easy.values[0])
        wsm_intermediate.append(row.intermediate.values[0])
        first_node = node

    weighted_path_gpd = gpd.GeoDataFrame({'fid': links, 'length': length, 'dpn': dpn, 'angle': angle,
                                          'surface_cost': surface_cost, 'total_time': total_time,
                                          'easy':wsm_easy,'intermediate':wsm_intermediate,'geometry': geom})
    return weighted_path_gpd


def smooth_linestring(path_gpd, smooth_sigma):
    geom = path_gpd['geometry'].tolist()
    multi_line = geometry.MultiLineString(geom)
    linestring = ops.linemerge(multi_line)
    start_coord = linestring.coords[0]
    end_coord = linestring.coords[-1]
    smooth_x = np.array(filters.gaussian_filter1d(
        linestring.xy[0],
        smooth_sigma)
    )
    smooth_y = np.array(filters.gaussian_filter1d(
        linestring.xy[1],
        smooth_sigma)
    )
    smoothed_coords = np.hstack((smooth_x, smooth_y))
    smoothed_coords = zip(smooth_x, smooth_y)
    smoothed_coords = [i for i in smoothed_coords]
    smoothed_coords[0] = start_coord
    smoothed_coords[-1] = end_coord
    linestring_smoothed = LineString(smoothed_coords)
    new_path_gpd = gpd.GeoSeries({'geometry': linestring_smoothed})
    return new_path_gpd


def plot_route(Haytor_map, study_area_shapely, dpn_routes):
    bounds = Haytor_map.bounds
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=crs.OSGB(approx=True))

    # display background map
    show(Haytor_map,ax=ax, extent=extent, zorder=0)

    # display path
    dpn_routes.plot(ax=ax, zorder=4, edgecolor='black', linewidth=0.7, label='DPN')

    # plot north arrow
    x, y, arrow_length = 0.1, 0.99, 0.06
    ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
                arrowprops=dict(facecolor='black', width=0.8, headwidth=3, headlength=2),
                ha='center', va='center', fontsize=4, xycoords=ax.transAxes)

    # plot scale bar
    scale1 = ScaleBar(dx=1, location='lower center', scale_loc='bottom', color='black', box_alpha=0.1
                      , font_properties={'size': 'xx-small'})

    ax.add_artist(scale1)
    # plot legend
    plt.legend(loc='lower right', fontsize=3)
    # set the extent to the study area
    display_extent = ((study_area_shapely.bounds[0] - 10, study_area_shapely.bounds[2] + 10,
                       study_area_shapely.bounds[1] - 10, study_area_shapely.bounds[3] + 10))

    ax.set_extent(display_extent, crs=crs.OSGB(approx=True))
    plt.savefig('dpn_al_1681_int_dpn1')
    plt.show()


def plot_routes(Haytor_map, study_area_shapely, easy_routes, int_routes, chal_routes):
    bounds = Haytor_map.bounds
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=crs.OSGB(approx=True))

    # display background map
    show(Haytor_map,ax=ax, extent=extent, zorder=0)

    # display path
    easy_routes.plot(ax=ax, zorder=7, edgecolor='green', linewidth=0.7, label='Easy')
    int_routes.plot(ax=ax, zorder=6, edgecolor='orange', linewidth=0.7, label='Intermediate')
    chal_routes.plot(ax=ax, zorder=5, edgecolor='red', linewidth=0.7, label='Challenging')

    # plot north arrow
    x, y, arrow_length = 0.1, 0.99, 0.06
    ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
                arrowprops=dict(facecolor='black', width=0.8, headwidth=3, headlength=2),
                ha='center', va='center', fontsize=4, xycoords=ax.transAxes)

    # plot scale bar
    scale1 = ScaleBar(dx=1, location='lower center', scale_loc='bottom', color='black', box_alpha=0.1
                      , font_properties={'size': 'xx-small'})

    ax.add_artist(scale1)
    # plot legend inside
    plt.legend(loc='lower right', fontsize=3)
    # plot legend outside
    # plt.legend(loc='upper left', fontsize=3, bbox_to_anchor=(1.02, 1))
    # set the extent to the study area
    display_extent = ((study_area_shapely.bounds[0] - 10, study_area_shapely.bounds[2] + 10,
                       study_area_shapely.bounds[1] - 10, study_area_shapely.bounds[3] + 10))

    ax.set_extent(display_extent, crs=crs.OSGB(approx=True))
    plt.savefig('weighted_al_1681_int_dpn1')
    plt.show()


def create_weighted_routes(study_area_shapely, SX77_map, points, network_links, network_nodes):
    weighted_path_dpn, weighted_path_surface_cost, graph = create_graph_and_paths(network_links, points)

    weighted_path_easy, weighted_path_intermediate, weighted_path_difficult, graph = create_weighted_graph_and_paths(
        network_links, points)

    weighted_path_dpn_gpd = create_path_gpd(weighted_path_dpn, network_links, graph)
    weighted_path_easy_gpd = create_path_gpd(weighted_path_easy, network_links, graph)
    weighted_path_intermediate_gpd = create_path_gpd(weighted_path_intermediate, network_links, graph)
    weighted_path_difficult_gpd = create_path_gpd(weighted_path_difficult, network_links, graph)

    linestring_smoothed_dpn = smooth_linestring(weighted_path_dpn_gpd, 1)
    linestring_smoothed_easy = smooth_linestring(weighted_path_easy_gpd, 1)
    linestring_smoothed_intermediate = smooth_linestring(weighted_path_intermediate_gpd, 1)
    linestring_smoothed_difficult = smooth_linestring(weighted_path_difficult_gpd, 1)

    plot_route(SX77_map, study_area_shapely,linestring_smoothed_dpn)
    plot_routes(SX77_map, study_area_shapely,linestring_smoothed_easy, linestring_smoothed_intermediate,
                linestring_smoothed_difficult)
    return weighted_path_easy_gpd, weighted_path_intermediate_gpd, weighted_path_difficult_gpd, \
           linestring_smoothed_easy, linestring_smoothed_intermediate, linestring_smoothed_difficult


def create_haytor_routes(OS_National_Grids, SX77_map):
    study_area_shapely = OS_National_Grids[OS_National_Grids['tile_name'] == "SX7677"].geometry.cascaded_union

    network_nodes = gpd.read_file(
        os.path.join('../Study_area', 'SX7677', 'Final Networks', 'network_nodes_dpn_3.geojson'))

    network_links = gpd.read_file(
        os.path.join('../Study_area', 'SX7677', 'Final Networks', 'network_links_with_weights.geojson'))

    # Points for Haytor

    # from Haytor rocks to B3387
    # points = ['int_dpn1', 'dpn_1127']
    # points = ['int_dpn1', 'dpn_449']
    # points = ['int_dpn1', 'int_dpn28']
    # points = ['dpn_32', 'dpn_58']


    # from top_righ to haytor rocks path
    points = ['al_1681', 'int_dpn1']

    weighted_path_easy_gpd, weighted_path_intermediate_gpd, weighted_path_difficult_gpd, linestring_smoothed_easy, linestring_smoothed_intermediate, linestring_smoothed_difficult = \
        create_weighted_routes(study_area_shapely, SX77_map, points, network_links, network_nodes)

def main():
    OS_National_Grids = gpd.read_file(
        os.path.join('../OS-British-National-Grids-main', 'OS-British-National-Grids-main', 'os_bng_grids.gpkg'),
        layer='1km_grid')

    SX77_map = rasterio.open(
        os.path.join('../MasterMap', 'Download_SX7677_Masterma_Vectormap_Raster_2099357', 'mastermap_1_to_1000_4732477', 'sx', 'sx77', 'sx7677.tif'))

    create_haytor_routes(OS_National_Grids, SX77_map)

if __name__ == "__main__":
    main()
