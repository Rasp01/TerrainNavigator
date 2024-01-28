 # packages needed
import shapely.affinity
from shapely.geometry import Point, mapping, LineString
import numpy as np
import rasterio
from rasterio import plot, mask
import geopandas as gpd
import os
import matplotlib.pyplot as plt
from cartopy import crs
from math import atan, degrees
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# functions


def limit_elevation(elevation, study_area_shapely):
    elevation_study_area = shapely.affinity.scale(study_area_shapely, xfact=1.1, yfact=1.1, origin='center')
    study_area = mapping(elevation_study_area)
    elevation_mask, transform_index = mask.mask(elevation, [study_area], filled=False, crop=False)
    return elevation_mask, transform_index


def create_nodes(study_area_shapely, elevation_mask, transform_index,spider_grid_value):
    count_x = np.arange(study_area_shapely.bounds[0], study_area_shapely.bounds[2] + spider_grid_value, spider_grid_value, dtype=int)
    count_y = np.arange(study_area_shapely.bounds[1], study_area_shapely.bounds[3] + spider_grid_value, spider_grid_value, dtype=int)

    points = []
    height = []
    for i in count_x:
        for j in count_y:
            points.append(Point(i, j))
            point_index = rasterio.transform.rowcol(transform_index, i,
                                                    j)
            point_elevation = elevation_mask[0][point_index]
            height.append(point_elevation)
    nodes_gpd = gpd.GeoSeries(points)

    G = nodes_gpd.geometry.apply(lambda geom: geom.wkb)
    nodes_gpd = nodes_gpd.loc[G.drop_duplicates().index]

    network_nodes = gpd.GeoDataFrame(geometry=nodes_gpd)
    network_nodes['fid'] = range(1, len(network_nodes) + 1)
    network_nodes['fid'] = 'al_' + network_nodes['fid'].astype(str)
    network_nodes['height'] = height
    return network_nodes, count_x, count_y


def calculate_climb_time(start_height, end_height, climb_time_forward):
    # get the angle of elevation change and climb time

    if start_height > end_height:
        change_height = start_height - end_height
        climb_time_forward.append(0)
        # for reverse direction
        climb_time_forward.append(change_height / 10)
    else:
        change_height = end_height - start_height
        climb_time_forward.append(change_height / 10)
        # for reverse direction
        climb_time_forward.append(0)

    return change_height, climb_time_forward


def calculate_steepness(change_height, line_length, angles):
    angle = degrees(atan((change_height / line_length)))
    angles.extend([angle, angle])

    return angles


def create_link(row, line, lines, start_node, end_node, length, angles, climb_time_forward):
    start_node.append(row.fid)
    point = Point(line.coords[-1])
    end_point_row = network_nodes.loc[network_nodes['geometry'] == point]
    end_node.append(end_point_row.fid.values[0])
    start_node.append(end_point_row.fid.values[0])
    end_node.append(row.fid)

    # get the line geometry
    lines.extend([line, line])

    # get the length of the line
    length.extend([line.length, line.length])

    # get the angle of elevation change and climb time
    change_height, climb_time_forward = calculate_climb_time(row.height,
                                                             end_point_row.height.values[0], climb_time_forward)

    # print(change_height)
    angles = calculate_steepness(change_height, line.length, angles)

    return lines, start_node, end_node, length, angles, climb_time_forward


def create_links(network_nodes, count_x, count_y,spider_grid_value):
    lines = []
    start_node = []
    end_node = []
    length = []
    angles = []
    climb_time_forward = []
    for index, row in network_nodes.iterrows():
        if row.geometry.y == np.amin(count_y) and row.geometry.x != np.amax(count_x):
            line = LineString([(row.geometry.x, row.geometry.y), (row.geometry.x + spider_grid_value, row.geometry.y)])
            lines, start_node, end_node, length, angles, climb_time_forward = create_link(row, line, lines, start_node,
                                                                                          end_node, length, angles,
                                                                                          climb_time_forward)

            line = LineString([(row.geometry.x, row.geometry.y), (row.geometry.x + spider_grid_value, row.geometry.y + spider_grid_value)])
            lines, start_node, end_node, length, angles, climb_time_forward = create_link(row, line, lines, start_node,
                                                                                          end_node, length, angles,
                                                                                          climb_time_forward)

            line = LineString([(row.geometry.x, row.geometry.y), (row.geometry.x, row.geometry.y + spider_grid_value)])
            lines, start_node, end_node, length, angles, climb_time_forward = create_link(row, line, lines, start_node,
                                                                                          end_node, length, angles,
                                                                                          climb_time_forward)

        elif row.geometry.x != np.amax(count_x) and row.geometry.y == np.amax(count_y):
            line = LineString([(row.geometry.x, row.geometry.y), (row.geometry.x + spider_grid_value, row.geometry.y)])
            lines, start_node, end_node, length, angles, climb_time_forward = create_link(row, line, lines, start_node,
                                                                                          end_node, length, angles,
                                                                                          climb_time_forward)

            line = LineString([(row.geometry.x, row.geometry.y), (row.geometry.x + spider_grid_value, row.geometry.y - spider_grid_value)])
            lines, start_node, end_node, length, angles, climb_time_forward = create_link(row, line, lines, start_node,
                                                                                          end_node, length, angles,
                                                                                          climb_time_forward)

        elif row.geometry.x == np.amax(count_x) and row.geometry.y != np.amax(count_y):
            line = LineString([(row.geometry.x, row.geometry.y), (row.geometry.x, row.geometry.y + spider_grid_value)])
            lines, start_node, end_node, length, angles, climb_time_forward = create_link(row, line, lines, start_node,
                                                                                          end_node, length, angles,
                                                                                          climb_time_forward)

        elif row.geometry.x != np.amax(count_x) and row.geometry.y != np.amax(count_y):
            line = LineString([(row.geometry.x, row.geometry.y), (row.geometry.x + spider_grid_value, row.geometry.y - spider_grid_value)])
            lines, start_node, end_node, length, angles, climb_time_forward = create_link(row, line, lines, start_node,
                                                                                          end_node, length, angles,
                                                                                          climb_time_forward)

            line = LineString([(row.geometry.x, row.geometry.y), (row.geometry.x + spider_grid_value, row.geometry.y)])
            lines, start_node, end_node, length, angles, climb_time_forward = create_link(row, line, lines, start_node,
                                                                                          end_node, length, angles,
                                                                                          climb_time_forward)

            line = LineString([(row.geometry.x, row.geometry.y), (row.geometry.x + spider_grid_value, row.geometry.y + spider_grid_value)])
            lines, start_node, end_node, length, angles, climb_time_forward = create_link(row, line, lines, start_node,
                                                                                          end_node, length, angles,
                                                                                          climb_time_forward)

            line = LineString([(row.geometry.x, row.geometry.y), (row.geometry.x, row.geometry.y + spider_grid_value)])
            lines, start_node, end_node, length, angles, climb_time_forward = create_link(row, line, lines, start_node,
                                                                                          end_node, length, angles,
                                                                                          climb_time_forward)

    links_fid = range(1, len(lines) + 1)
    network_links = gpd.GeoDataFrame({'fid': links_fid, 'startnode': start_node,
                                      'endnode': end_node, 'length': length, 'angle': angles,
                                      'climb_time_forward': climb_time_forward, 'geometry': lines})
    # network_links = gpd.GeoDataFrame({'fid':links_fid,'startNodes':start_node,'geometry':lines})
    return network_links


def plot_nodes(background_map, study_area_shapely, elevation_mask, transform_index, network_nodes):
    back_array = background_map.read(1)
    palette = np.array([value for key, value in background_map.colormap(1).items()])
    background_image = palette[back_array]
    bounds = background_map.bounds
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=crs.OSGB(approx=True))

    # display background map
    ax.imshow(background_image, origin='upper', extent=extent, zorder=0)

    # display elevation
    # rasterio.plot.show(elevation_mask, alpha=0.6, transform=transform_index, ax=ax, zorder=1,
    #                    cmap='terrain')

    # displaying nodes
    network_nodes.plot(ax=ax, zorder=3, markersize=0.2,label="network nodes")

    # set the extent to the study area
    display_extent = ((study_area_shapely.bounds[0] - 100, study_area_shapely.bounds[2] + 100,
                       study_area_shapely.bounds[1] - 100, study_area_shapely.bounds[3] + 100))

    ax.set_extent(display_extent, crs=crs.OSGB(approx=True))


def plot_links(background_map, study_area_shapely, elevation_mask, transform_index, network_nodes, network_links):
    back_array = background_map.read(1)
    palette = np.array([value for key, value in background_map.colormap(1).items()])
    background_image = palette[back_array]
    bounds = background_map.bounds
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    fig = plt.figure(figsize=(3, 3), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=crs.OSGB(approx=True))

    # display background map
    ax.imshow(background_image, origin='upper', extent=extent, zorder=0)

    # display elevation
    # rasterio.plot.show(elevation_mask, alpha=0.6, transform=transform_index, ax=ax, zorder=1,
    #                    cmap='terrain')

    # displaying nodes
    network_nodes.plot(ax=ax, zorder=3, markersize=0.2,label="network nodes")

    # displaying links
    network_links.plot(ax=ax, zorder=2, edgecolor='blue', linewidth=0.2,label="network links")

    # set the extent to the study area
    display_extent = ((study_area_shapely.bounds[0] - 100, study_area_shapely.bounds[2] + 100,
                       study_area_shapely.bounds[1] - 100, study_area_shapely.bounds[3] + 100))

    ax.set_extent(display_extent, crs=crs.OSGB(approx=True))
    plt.show()


# files to import

OS_National_Grids = gpd.read_file(
    os.path.join('../OS-British-National-Grids-main', 'OS-British-National-Grids-main', 'os_bng_grids.gpkg'),
    layer='1km_grid')

study_area_shapely = OS_National_Grids[OS_National_Grids['tile_name'] == "SX7478"].geometry.unary_union
# study_area_shapely = OS_National_Grids[OS_National_Grids['tile_name'] == "SX7677"].geometry.unary_union

SX77_map = rasterio.open(
    os.path.join('../OS Explorer Maps', 'Download_SX77-Haytor_2033809', 'raster-25k_4596071', 'sx', 'sx77.tif'))

elevation = rasterio.open(
    os.path.join('../OS Elevation', 'SX77_elevation', 'terrain-5-dtm_4616587', 'sx', 'sx77ne_nw', 'w001001.adf'))


elevation_mask, transform_index = limit_elevation(elevation, study_area_shapely)

network_nodes, count_x, count_y = create_nodes(study_area_shapely, elevation_mask, transform_index,25)

plot_nodes(SX77_map, study_area_shapely, elevation_mask, transform_index, network_nodes)

network_links = create_links(network_nodes, count_x, count_y,25)

plot_links(SX77_map, study_area_shapely, elevation_mask, transform_index, network_nodes, network_links)

# network_links.to_file("Study_area/SX7677/Final Networks/network_links_al.geojson", driver='GeoJSON',crs='EPSG:27700')
# network_nodes.to_file("Study_area/SX7677/Final Networks/network_nodes_al.geojson", driver='GeoJSON',crs='EPSG:27700')

network_links.to_file("../Study_area/SX7478/network_links_al_2.geojson", driver='GeoJSON',crs='EPSG:27700')
network_nodes.to_file("../Study_area/SX7478/network_nodes_al_2.geojson", driver='GeoJSON',crs='EPSG:27700')
