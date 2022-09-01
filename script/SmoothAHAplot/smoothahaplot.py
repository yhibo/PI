import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from . import parameters
from .plot_style import PlotStyle, PlotUtil
from .wt_from_ex_files import ExNodeParser

from typing import Dict, List, Optional, Tuple, Union


def confirm_segment_number(func):
    def inner(self, *args, **kwargs):
        if not (self._n_segments == 17 or self._n_segments == 18):
            raise SegmentSizeError(
                f'Incorrect number of segment values provided: {self._n_segments}. Provide either '
                f'17 or 18 segment values')
        return func(self, *args, **kwargs)

    return inner


class AHASegmentalValues:

    def __init__(self, segments: Union[pd.Series, Dict]):
        self._segments = None
        self._n_segments = 0
        self.segments = segments
        self._segmental_values = []
        self._get_segmental_values()

    @property
    def segments(self):
        return self._segments

    @segments.setter
    def segments(self, new_segments: List[int]):
        if not (len(new_segments) == 17 or len(new_segments) == 18):
            raise SegmentSizeError(f'Incorrect number of segment values provided: {len(new_segments)}. Provide either '
                                   f'17 or 18 segment values')
        self._segments = new_segments
        self._n_segments = len(self._segments)

    @confirm_segment_number
    def _extract_values_from_dict(self):

        for segment_name in parameters.AHA_SEGMENT_NAMES[self._n_segments]:
            try:
                self._segmental_values.append(self.segments[segment_name])
            except KeyError:
                exit('Value of segment {} is missing. Provide values for all AHA 17/18 segments'.format(segment_name))

    @confirm_segment_number
    def _extract_values_from_series(self):

        segments_s = self.segments.reindex(parameters.AHA_SEGMENT_NAMES(self._n_segments), axis=1)
        if segments_s.isnull().values.any():
            exit('Segmental names are incompatibile:\n{}\n{}'.format(segments_s.index,
                                                                     parameters.AHA_SEGMENT_NAMES(self._n_segments)))
        self._segmental_values = segments_s.values

    @confirm_segment_number
    def _get_segmental_values(self):

        if isinstance(self.segments, dict):
            self._extract_values_from_dict()
        elif isinstance(self.segments, pd.Series):
            self._extract_values_from_series()
        else:
            raise SegmentsError(f'Unknown format of the segment list provided: {type(self.segments)}')


class AHAInterpolation:

    def __init__(self, segmental_values: List[int]):
        self._segmental_values = None
        self._n_segments = 17
        self.interpolated_values = None
        self.ip = None

        self.segmental_values = segmental_values
        self._assign_interpolation_parameters()

    @property
    def segmental_values(self):
        return self._segmental_values

    @confirm_segment_number
    @segmental_values.setter
    def segmental_values(self, new_values: List[int]):
        self._segmental_values = new_values
        self._n_segments = len(self._segmental_values)

    @confirm_segment_number
    def _assign_interpolation_parameters(self):
        if self._n_segments == 17:
            self.ip = parameters.AHA17Parameters()
        self.ip = parameters.AHA18Parameters()

    def _interpolate_directions(self, regional_values: List[int]) -> np.ndarray:
        res = self.ip.resolution[0]
        n_segments = len(regional_values)
        interpolated_array = np.zeros(self.ip.resolution[0])

        for i in range(n_segments):
            interpolated_array[int(res / n_segments) * i:int(res / n_segments * i + res / n_segments)] = \
                np.linspace(regional_values[i], regional_values[(i + 1) % n_segments], int(res / n_segments))

        return interpolated_array

    @staticmethod
    def _basal_mid(basal: np.ndarray, mid: np.ndarray) -> np.ndarray:
        """
        :return: Helper array for better basal segments visualization
        """
        return (basal * 3 + mid) / 4

    def _interpolate_17_aha_values_along_circle(self):
        """
        Interpolate the initial 17 values, to achieve smooth transition among segments.
        """
        basal = self._interpolate_directions(self.segmental_values[:6])
        mid = self._interpolate_directions(self.segmental_values[6:12])
        apex_mid = self._interpolate_directions(self.segmental_values[12:16])
        apex = np.repeat(self.segmental_values[16], self.ip.resolution[0])
        return basal, mid, apex_mid, apex

    def _interpolate_18_aha_values_along_circle(self):
        """
        Interpolate the initial 18 values, to achieve smooth transition among segments.
        """
        basal = self._interpolate_directions(self.segmental_values[:6])
        mid = self._interpolate_directions(self.segmental_values[6:12])
        apex_mid = self._interpolate_directions(self.segmental_values[12:])
        apex = np.repeat(np.sum(self.segmental_values[12:]) / 6, self.ip.resolution[0])
        return basal, mid, apex_mid, apex

    @confirm_segment_number
    def interpolate_aha_values(self):
        if self._n_segments == 17:
            interp_func = self._interpolate_17_aha_values_along_circle
        else:
            interp_func = self._interpolate_18_aha_values_along_circle

        # Set up the circular interpolation matrices
        basal, mid, apex_mid, apex = interp_func()
        along_x = np.array([basal, self._basal_mid(basal, mid), mid, apex_mid, apex])

        # Adjust to visualisation
        along_x = np.roll(along_x, int(self.ip.resolution[0] / 4), axis=1)
        along_x = np.flip(along_x, 0)

        # Interpolate along the radius
        f = interp1d(self.ip.plot_levels, along_x, axis=0)
        along_x_y = f(np.linspace(0, 1, self.ip.resolution[1]))

        return along_x_y


class AHAPlotting:
    """
    Class for producing smooth 17 and 18 segment left ventricle plot, recommended by American Heart Association:
        http://www.pmod.com/files/download/v34/doc/pcardp/3615.htm
    Inspired with the 'bullseye plot':
        https://matplotlib.org/gallery/specialty_plots/leftventricle_bulleye.html
    Available at:
        https://github.com/MaciejPMarciniak/smoothAHAplot

    Two helper methods are included, adjusted to plot myocardial work and strain values with proper scales.
    """

    def __init__(self, values: List[int], plot_output_path: str =''):

        self._segment_values: List[int] = None
        self._n_segments: int = None
        self._output_path: str = None
        self._theta = None

        self.ps = PlotStyle()
        self.pu = PlotUtil()
        self.ip = parameters.AHA17Parameters()

        self.segment_values = values
        self.output_path = plot_output_path
        self.theta = np.linspace(0, 2 * np.pi, self.ip.resolution[0])

        ai = AHAInterpolation(self.segment_values)
        self._interpolated_segment_values: np.ndarray = ai.interpolate_aha_values()

        self.fig, self.ax = plt.subplots(figsize=(12, 8), nrows=1, ncols=1, subplot_kw=dict(projection='polar'))
        self.levels = None

    @property
    def segment_values(self) -> List[int]:
        return self._segment_values

    @segment_values.setter
    def segment_values(self, values: List[int]):
        self._n_segments = len(values)
        self._segment_values = values

    @property
    def n_segments(self):
        return self._n_segments

    @property
    def theta(self) -> np.ndarray:
        return self._theta

    @theta.setter
    def theta(self, angles: np.ndarray):
        assert len(angles) == self.ip.resolution[0], ('Number of provided angle values {len(angles)} does not match the' 
                                                      ' desired resolution (self.ip.resolution[0]}')
        self._theta = angles

    @property
    def interpolated_segment_values(self):
        return self._interpolated_segment_values

    def _write_segment_names(self):
        for wall in range(len(parameters.AHA_SEGMENT_NAMES['walls'])):

            segment_name_direction = np.deg2rad(self.pu.annotation_shift_functions[
                len(parameters.AHA_SEGMENT_NAMES['walls'])](wall))
            segment_name_position = self.ip.radial_position[-1] + self.ps.positional_parameters['segment_names_position']
            segment_name = parameters.AHA_SEGMENT_NAMES['walls'][wall]
            segment_name_orientation = self.ps.positional_parameters['segment_name_orientations'][wall]

            self.ax.text(x=segment_name_direction, y=segment_name_position, s=segment_name,
                         rotation=segment_name_orientation, **self.ps.segment_name_style)

    def _draw_radial_bounds(self):
        for radial_bound in range(self.ps.aha_bounds[self.n_segments].shape[0]):
            self.ax.plot(self.theta, np.repeat(self.ps.aha_bounds[self.n_segments][radial_bound], self.theta.shape),
                         **self.ps.segment_border_style)

    def _draw_bounds(self, inner: float, outer: float, n_borders: int):
        assert 0 <= inner <= 1, f'Inner starting point value must be between 0 and 1 (is {inner})'
        assert 0 <= outer <= 1, f'Outer starting point value must be between 0 and 1 (is {outer})'
        assert inner < outer, f'Inner ({inner}) cannot be greater than outer ({outer})'
        assert n_borders == 4 or n_borders == 6, (f'Only 4 or 6 borders between segments are allowed ({n_borders} '
                                                  f'provided)')

        shift_function = self.pu.border_shift_functions[n_borders]

        for segment_border in range(n_borders):
            border_orientation = np.deg2rad(shift_function(segment_border))
            self.ax.plot([border_orientation, border_orientation], [inner, outer],
                         **self.ps.segment_border_style)

    def _draw_outer_bounds(self):
        self._draw_bounds(self.ps.aha_bounds[self._n_segments][1], 1, 6)

    @confirm_segment_number
    def _draw_inner_bounds(self):
        if self._n_segments == 17:
            self._draw_bounds(self.ps.aha_bounds[self._n_segments][0], self.ps.aha_bounds[self._n_segments][1], 4)
        else:
            self._draw_bounds(0, self.ps.aha_bounds[self._n_segments][0], 6)

    def _draw_aha_bounds(self):
        self._draw_radial_bounds()
        self._draw_outer_bounds()
        self._draw_inner_bounds()

    def _annotate_segment(self, angle: float, position: float, value: Union[int, float]):
        self.ax.text(angle, position, value, **self.ps.values_style)

    @staticmethod
    def _fix_negative_zero(value: Union[float, int]):
        return 0 if np.abs(np.round(value, 1)) < 0.1 else int(value)

    def _get_annotation_angle(self, n_level_segments: int, segment: int):
        return np.deg2rad(self.pu.annotation_shift_functions[n_level_segments](segment))

    def _annotate_basal_segments(self):
        n_level_segments = len(parameters.AHA_SEGMENT_NAMES['walls'])
        for segment in range(n_level_segments):
            angle = self._get_annotation_angle(n_level_segments, segment)
            position = float(np.mean([self.ps.aha_bounds[self.n_segments][-2], self.ps.aha_bounds[self.n_segments][-1]]))
            value = self._fix_negative_zero(self.segment_values[segment])
            self._annotate_segment(angle, position, value)

    def _annotate_mid_segments(self):
        n_level_segments = len(parameters.AHA_SEGMENT_NAMES['walls'])
        for segment in range(n_level_segments):
            angle = self._get_annotation_angle(n_level_segments, segment)
            position = float(np.mean([self.ps.aha_bounds[self.n_segments][-3], self.ps.aha_bounds[self.n_segments][-2]]))
            value = self._fix_negative_zero(self.segment_values[segment+6])
            self._annotate_segment(angle, position, value)

    def _annotate_apical_segments(self):
        if self._n_segments == 17:
            n_level_segments = 4
            for segment in range(n_level_segments):
                angle = self._get_annotation_angle(n_level_segments, segment)
                position = float(np.mean([self.ps.aha_bounds[self.n_segments][0], self.ps.aha_bounds[self.n_segments][1]]))
                value = self._fix_negative_zero(self.segment_values[segment + 12])
                self._annotate_segment(angle, position, value)

            angle = position = 0
            value = self._fix_negative_zero(self.segment_values[-1])
            #self._annotate_segment(angle, position, value)
        else:
            n_level_segments = len(parameters.AHA_SEGMENT_NAMES['walls'])
            for segment in range(n_level_segments):
                angle = self._get_annotation_angle(n_level_segments, segment)
                position = float(np.mean([0, self.ps.aha_bounds[self.n_segments][0]]))
                value = self._fix_negative_zero(self.segment_values[segment + 12])
                self._annotate_segment(angle, position, value)

    def _annotate_aha_segments(self):
        self._annotate_basal_segments()
        self._annotate_mid_segments()
        self._annotate_apical_segments()

    def _clear_bullseye_plot(self):
        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])
        self.ax.set_ylim([0, 1])

    def _normalize_data(self, norm: Optional[Union[BoundaryNorm, Tuple[int, int]]] = None):
        if norm is None:
            norm = mpl.colors.Normalize(vmin=self.seg_values.min(), vmax=self.seg_values.max())
        elif isinstance(norm, tuple) and len(norm) == 2:
            norm = mpl.colors.Normalize(vmin=norm[0], vmax=norm[1])
        print(norm)
        return norm

    def _add_color_bar(self, units: str = 'Units', cmap: str = 'jet', norm = None):
        bar = self.fig.add_axes([0.05, 0.1, 0.2, 0.05])
        cb1 = mpl.colorbar.ColorbarBase(bar, cmap=cmap, norm=norm, orientation='horizontal')
        cb1.set_label(units, size=16)
        cb1.ax.tick_params(labelsize=14, which='major')

    def _color_plot(self, cmap: str = 'jet', norm=None, smooth_contour=False):
        extended_radial_position = np.repeat(self.ip.radial_position[:, np.newaxis], self.ip.resolution[0], axis=1).T
        extended_radial_angle = np.repeat(self.theta[:, np.newaxis], extended_radial_position.shape[1], axis=1)
        ravelled_segment_values = np.array(self.interpolated_segment_values).T
        # Color the plot
        if smooth_contour:
            levels = MaxNLocator(nbins=12).tick_values(-30, 30)
            cf = self.ax.contourf(extended_radial_angle, extended_radial_position, ravelled_segment_values, cmap=cmap,
                                  levels=levels)
            cf.axes.grid(None)
        else:
            self.ax.pcolormesh(extended_radial_angle, extended_radial_position, ravelled_segment_values, cmap=cmap,
                               norm=norm)

    def bullseye_17_smooth(self, cmap: str ='jet', norm: Optional[Union[BoundaryNorm, Tuple[int, int]]] = None,
                           title: str ='Smooth 17 AHA plot', smooth_contour: bool =False, units: str = ''):
        """
        Function to create the smooth representation of the AHA 17 segment plot
        :param cmap: ColorMap or None, optional
            Optional argument to set the desired colormap
        :param norm: tuple, BoundaryNorm or None
            Tuple (vmin, vmax) - normalize data into the [0.0, 1.0] range with minimum and maximum desired values.
        :param units: str
            Label of the color bar
        :param title: str
            Title of the plot
        :param smooth_contour: Bool
            Whether to smooth the plot. Useful for level-based color map
        :return fig: matplotlib.pyplot.figure
            The figure on which the 17 AHA plot has been drawn
        """

        self._draw_aha_bounds()
        self._write_segment_names()
        self._clear_bullseye_plot()

        normalized_data = self._normalize_data(norm)
        self._color_plot(cmap, normalized_data, smooth_contour)
        self._add_color_bar(units, cmap, normalized_data)

        self._annotate_aha_segments()

        self.ax.set_title(title, fontsize=24, pad=40)

        return self.fig

    """
    def bullseye_18_smooth(self, fig, ax, cmap='jet', color_seg_names=False, norm=None, units='Units',
                           title='Smooth 18 AHA plot', smooth_contour=False, echop=False):
        
        Function to create the smooth representation of the AHA 18 segment plot
        :param fig: matplotlib.pyplot.figure
            The plot is drawn on this figure
        :param ax: matplotlib.pyplot.axes
            Axes of the figure, for 18 AHA and colorbar.
        :param cmap: ColorMap or None, optional
            Optional argument to set the desired colormap
        :param color_seg_names: boolean, default - False
            Whether or not to color the segment names with traditional echocardiography colors
        :param norm: tuple, BoundaryNorm or None
            Tuple (vmin, vmax) - normalize data into the [0.0, 1.0] range with minimum and maximum desired values.
        :param units: str
            Label of the color bar
        :param title: str
            Title of the plot
        :param smooth_contour:
            Whether to smooth the plot. Useful for level-based color map
        :param echop: Bool
            If true, the resultant plot is structured as the 18 AHA plots in GE EchoPAC (TM)
        :return fig: matplotlib.pyplot.figure
            The figure on which the 18 AHA plot has been drawn
       

        if norm is None:
            norm = mpl.colors.Normalize(vmin=self.seg_values.min(), vmax=self.seg_values.max())
        elif isinstance(norm, tuple) and len(norm) == 2:
            norm = mpl.colors.Normalize(vmin=norm[0], vmax=norm[1])
        else:
            pass

        rot = [0, 60, -60, 0, 60, -60]
        seg_align = 90
        seg_names = parameters.AHA_SEGMENT_NAMES
        seg_names_pos = np.repeat([0.06], 6)

        # -----Basic assumptions on resolution--------------------------------------------------------------------------
        self.seg_values = np.array(self.seg_values).ravel()

        if echop:
            theta -= np.pi / 3
        r =
        # ==============================================================================================================

        # -----Drawing bounds of the plot-------------------------------------------------------------------------------
        linewidth = 2
        # Create the radial bounds
        for i in range(r.shape[0]):
            ax.plot(theta, np.repeat(r[i], theta.shape), '-k', lw=linewidth)

        # Create the bounds for the segments 1-18
        for i in range(6):
            theta_i = np.deg2rad(i * 60)
            ax.plot([theta_i, theta_i], [0, 1], '-k', lw=linewidth)
        # ==============================================================================================================

        # -----Linear interpolation-------------------------------------------------------------------------------------
        if len(self.seg_values) == 17:
            interp_data = self._interpolate_17_aha_values(self.seg_values)
        elif len(self.seg_values) == 18:
            interp_data = self._interpolate_18_aha_values(self.seg_values)
        else:
            exit('Wrong number of segments provided: {}'.format(len(self.seg_values)))
            return -1

        r = np.linspace(0, 1, interp_data.shape[0])
        # ==============================================================================================================

        # -----Fill segments 1:18---------------------------------------------------------------------------------------
        r0 = r
        r0 = np.repeat(r0[:, np.newaxis], self.ip.resolution[0], axis=1).T
        theta0 = theta
        theta0 = np.repeat(theta0[:, np.newaxis], r0.shape[1], axis=1)
        z = interp_data
        z = z.T

        # Annotate
        for i in range(6):
            ax.text(np.deg2rad(i * 60) + np.deg2rad(seg_align), 0.84,
                    0 if np.abs(np.round(self.seg_values[i], 1)) < 0.1 else  # condition to not allow 'negative 0'
                    int(self.seg_values[i]),
                    **self.ps.values_style)
            ax.text(np.deg2rad(i * 60) + np.deg2rad(seg_align), 0.55,
                    0 if np.abs(np.round(self.seg_values[i + 6], 1)) < 0.1 else  # condition to not allow 'negative 0'
                    int(self.seg_values[i + 6]),
                    **self.ps.values_style)
            ax.text(np.deg2rad(i * 60) + np.deg2rad(seg_align), 0.25,
                    0 if np.abs(np.round(self.seg_values[i + 12], 1)) < 0.1 else  # condition to not allow 'negative 0'
                    int(self.seg_values[i + 12]),
                    **self.ps.values_style)
            # Segment names
            ax.text(np.deg2rad(i * 60) + np.deg2rad(seg_align), r[-1] + seg_names_pos[i], seg_names[i],
                    fontsize=20, ha='center', va='center', rotation=rot[i],
                    color=self.ps.segment_name_colors[i] if color_seg_names else 'k',
                    path_effects=[pef.Stroke(linewidth=1, foreground='k'), pef.Normal()] if color_seg_names else None)
        # Colour
        if smooth_contour and (self.levels is not None):
            cf = ax.contourf(theta0, r0, z, cmap=cmap, levels=self.levels)
            cf.ax.axis('off')
        else:
            ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm)
        # ==============================================================================================================

        # -----Add plot features----------------------------------------------------------------------------------------
        # Create the axis for the colorbars
        bar = fig.add_axes([0.05, 0.1, 0.2, 0.05])
        cb1 = mpl.colorbar.ColorbarBase(bar, cmap=cmap, norm=norm, orientation='horizontal')
        cb1.set_label(units, size=16)
        cb1.ax.tick_params(labelsize=14, which='major')

        # Clear the bullseye plot
        ax.set_ylim([0, 1])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title(title, fontsize=24)
        # ==============================================================================================================

        return fig
 """

    def _plot_setup(self, data):

        if data is not None:
            assert len(data) == self._n_segments, 'Please provide correct number of segmental values for the plot. ' \
                                                 'len(data) = {}, n_segments =  {}'.format(len(data), self._n_segments)
            if not isinstance(data, list):
                data = self._get_segmental_values(data)

        return data

    def plot_myocardial_work(self, filename='', data=None, echop=False):

        self.seg_values = self._plot_setup(data)

        cmap = plt.get_cmap('rainbow')
        norm = (1000, 3000)

        if self._n_segments == 18:
            fig = self.bullseye_18_smooth(fig=fig, ax=ax, cmap=cmap, norm=norm, title='Myocardial work index',
                                          units='mmHg%', smooth_contour=False, echop=echop)
        else:
            fig = self.bullseye_17_smooth(fig=fig, ax=ax, cmap=cmap, norm=norm, title='Myocardial work index',
                                          units='mmHg%', smooth_contour=False, echop=echop)
        fig.savefig(os.path.join(self.output_path, filename))

    def plot_strain(self, filename='', data=None, echop=False):

        self.seg_values = self._plot_setup(data)

        self.levels = MaxNLocator(nbins=12).tick_values(-30, 30)
        cmap = plt.get_cmap('seismic_r')
        norm = BoundaryNorm(self.levels, ncolors=cmap.N, clip=True)
        fig, ax = plt.subplots(figsize=(12, 8), nrows=1, ncols=1, subplot_kw=dict(projection='polar'))
        if self._n_segments == 18:
            fig = self.bullseye_18_smooth(fig=fig, ax=ax, cmap=cmap, norm=norm, title='Longitudinal strain', units='%',
                                          smooth_contour=True, echop=echop)
        else:
            fig = self.bullseye_17_smooth(fig=fig, ax=ax, cmap=cmap, norm=norm, title='Longitudinal strain', units='%',
                                          smooth_contour=True, echop=echop)
        fig.savefig(os.path.join(self.output_path, filename))


class SegmentsError(AttributeError):
    """An error related to AHA segments"""


class SegmentSizeError(SegmentsError):
    """An error related to number of segments"""


if __name__ == '__main__':
    group_a = 2
    scale = 72
    exnode1 = r'G:\GenerationR\AtlasOutputLVLrv\ModesResTime\MeanEigen{}Scale{}.exnode'.format(group_a, scale)
    exnode2 = r'G:\GenerationR\AtlasOutputLVLrv\ModesResTime\MeanEigen{}Scalen{}.exnode'.format(group_a, scale)
    wt = ExNodeParser(exnode1, r'G:')
    wt.calc_wall_thickness()
    wt2 = ExNodeParser(exnode2, r'C:')
    wt2.calc_wall_thickness()
    wt_diff = wt.wt_difference(exnode2)
    plot_wt = SmoothAHAPlot(wt.wall_thicknesses, r'G:\GenerationR\AtlasOutputLVLrv\ModesResTime', n_segments=17)
    plot_wt.plot_wall_thickness(filename='WT_plot_{}_pos'.format(group_a))

    #
    plot_wt = SmoothAHAPlot(wt_diff, r'G:\GenerationR\AtlasOutputLVLrv\ModesResTime', n_segments=17)
    plot_wt.plot_wt_difference(filename='WT_pos_neg_difference_mode_{}'.format(group_a))
    plot_wt = SmoothAHAPlot(wt2.wall_thicknesses, r'G:\GenerationR\AtlasOutputLVLrv\ModesResTime', n_segments=17)
    plot_wt.plot_wall_thickness(filename='WT_plot_{}_neg'.format(group_a))