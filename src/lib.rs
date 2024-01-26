//! This crate provides functionality to apply Principal Coordinate Analysis.
//!
//! # Quick Start
//!
//! A very brief introduction into Principal Coordinates Analysis.
//!
//! ## What is Principal Coordinates Analysis (PCoA)
//! PCoA is a statistical method that turns distance data between items into a map-like visualization. It helps to reveal
//! which items are close to each other and which ones are far away. This visualization can also assist in identifying
//! groups or clusters among the items.
//!
//! ## Principal Coordinate Analysis and Multidimensional Scaling
//! Multidimensional Scaling is a family of statistical methods that focus on creating mappings of items based on distance.
//! Principal Coordinate Analysis is one type of Multidimensional Scaling which specifically works with numerical distances,
//! in which there is no measurement error — you have one precise distance measure for each pair of items.
//!
//! ## Principal Coordinate Analysis nad Principal Component Analysis
//! PCoA and Principal Component Analysis (PCA) are often confused due to their shared initials and both involving
//! dimensionality reduction. However, they differ in their primary objectives:
//! - PCA concentrates on shared variance, aiming to summarize multiple variables into the fewest components while maximizing
//!   the explanation of variance in each component.
//! - PCoA emphasizes distances and strives to identify dimensions that capture the greatest distances between items.
//!
//!
//! For more information, please read [an excellent article](https://towardsdatascience.com/principal-coordinates-analysis-cc9a572ce6c).
//!
//! ## Implementation
//!
//! This crate's implementation is based on [scikit-bio](https://scikit.bio/) implementation, however, most of performance
//! optimizations are not applied here.
//!
//! # Usage Example
//!
//! This section explains how to use the library.
//!
//! ## Brief
//!
//! Let's assume we have three points A, B, C with the respective distance matrix of 3x3 shape:
//!
//! |    |    |    |
//! |----|----|----|
//! |  0 | AB | AC |
//! | BA |  0 | BC |
//! | CA | CB |  0 |
//!
//! To call the `PCoA` function, you need to construct `DMatrix` instance from the distance matrix.
//! The following pseudo code demonstrates expected layout of one dimensional array which will be used later to the function:
//!
//! ```text
//! DMatrix::from_column_slice(3, 3, [0,AB,AC,BA,0,BC,CA,CB,0])
//! ```
//! `DMatrix` is a type from `nalgebra` crate, other types from this crate can be imported with `pcoa::nalgebra`.
//!
//! __Please note__ that current implementation assumes a symmetric matrix, so the following equalities are expected to be
//! hold: `AB=BA`, `AC=CA` and `BC=CB`.
//!
//! As a second argument, you need to pass dimensionality of principal coordinates. Typically, it equals to 2.
//!
//! As result, another `DMatrix` instance will be returned and it represents principal coordinates which can be used to
//! plot original items on a map.
//!
//! ## Code example
//!
//! A minimalistic example:
//!
//! ```
//! use pcoa::apply_pcoa;
//! use pcoa::nalgebra::DMatrix;
//!
//! // here, we have interest in only two coordinates (e.g. x and y)
//! let number_of_dimensions = 2;
//! // create a distance matrix from raw data. Matrix is expected to be symmetric with 3x3 shape
//! let distance_matrix = DMatrix::from_column_slice(3, 3, &[0_f64, 250., 450., 250., 0., 300., 450., 300., 0.]);
//! // apply pcoa
//! let coords_matrix = apply_pcoa(distance_matrix, number_of_dimensions).expect("cannot apply PCoA");
//!
//! // NOTE: transpose matrix to get first column for x coordinates and the second - for y coordinates.
//! let coords_matrix = coords_matrix.transpose();
//! let xs: Vec<_> = coords_matrix.column(0).iter().copied().collect();
//! let ys: Vec<_> = coords_matrix.column(1).iter().copied().collect();
//! // these are our coordinates
//! assert_eq!((xs[0].round(), ys[0].round()), (213., -60.));
//! assert_eq!((xs[1].round(), ys[1].round()), (24., 104.));
//! assert_eq!((xs[2].round(), ys[2].round()), (-237., -44.));
//! ```
//!
//! Here, we have the following distance matrix as input:
//!
//! |     |     |     |
//! |-----|-----|-----|
//! |  0  | 250 | 450 |
//! | 250 |  0  | 300 |
//! | 450 | 300 |  0  |
//!
//! As result, we get the following coordinates (rounded in test and here):
//!
//! ```text
//! A: (213,  -60)
//! B: ( 24,  104)
//! C: (-237, -44)
//! ```
//!
//! These coordinates retain original distances between points (with some precision loss) and can be used to visualize
//! original data in 2D space.
//!

// Reexport nalgebra to allow easy manipulation with its types
pub use nalgebra;

use nalgebra::{DMatrix, Dyn, OMatrix, OVector, Scalar};
use simba::scalar::{ComplexField, Field, SupersetOf};

/// Applies Principal Coordinates Analysis on given distance (dissimilarity) matrix.
/// Returns principal coordinates as a matrix which has the following orientation: number of rows is equal to
/// `number_of_dimensions` argument and number of columns is equal to the `distance_matrix` dimensionality (e.g. amount of data points).
pub fn apply_pcoa<S>(distance_matrix: DMatrix<S>, number_of_dimensions: usize) -> Option<DMatrix<S>>
where
    S: Scalar + Field + SupersetOf<f64> + Default + Copy + ComplexField,
{
    // center distance matrix, a requirement for PCoA here
    let centered_matrix = center_distance_matrix(distance_matrix);

    // perform eigen decomposition
    let (eigvals, eigvecs) = symmetric_eigen_decomposition(centered_matrix);

    // get principal coordinates from eigen vecs/vals
    get_principal_coordinates(eigvals, eigvecs, number_of_dimensions)
}

fn center_distance_matrix<S>(mut matrix: DMatrix<S>) -> DMatrix<S>
where
    S: Scalar + Field + SupersetOf<f64> + Default + Copy,
{
    let factor = S::from_subset(&-2.);
    mapv_inplace(&mut matrix, |val, _, _| (val * val).div(factor));

    let row_means = matrix.row_mean();
    let col_means = matrix.column_mean();
    let matrix_mean = matrix.mean();

    mapv_inplace(&mut matrix, |val, row, col| {
        val - row_means[col] - col_means[row] + matrix_mean
    });

    matrix
}

/// Calculates eigen decomposition for symmetric matrix.
/// Returns eigenvalues and eigenvectors, sorted in descending order.
fn symmetric_eigen_decomposition<S>(
    matrix: DMatrix<S>,
) -> (OVector<S::RealField, Dyn>, OMatrix<S, Dyn, Dyn>)
where
    S: Scalar + Field + SupersetOf<f64> + Default + Copy + ComplexField,
{
    let decomposition = matrix.symmetric_eigen();

    let mut eigvals = decomposition.eigenvalues;
    let mut eigvecs = decomposition.eigenvectors.transpose();

    // NOTE: sort eigenvalues and eigenvectors in descending order
    // TODO: this is simple buble sort, do it in a nicer/faster way
    let n = eigvals.len();
    for i in 0..n {
        for j in 0..(n - i - 1) {
            if eigvals[j] < eigvals[j + 1] {
                eigvals.swap((j, 0), (j + 1, 0));
                eigvecs.swap_rows(j, j + 1);
            }
        }
    }

    (eigvals, eigvecs)
}

fn get_principal_coordinates<S>(
    mut eigvals: OVector<S::RealField, Dyn>,
    mut eigvecs: OMatrix<S, Dyn, Dyn>,
    number_of_dimensions: usize,
) -> Option<DMatrix<S>>
where
    S: Scalar + Field + SupersetOf<f64> + Default + Copy + ComplexField,
{
    let n = eigvals.len();

    if number_of_dimensions > n {
        return None;
    }

    // return the coordinates that have a negative eigenvalue as 0
    // so reset negative values in eigenvalues and eigenvectors to zero
    let zero = S::from_subset(&0.);
    let zero_real = zero.real();
    let position = eigvals.iter().rposition(|v| v > &zero_real)?;

    if position < n - 1 {
        let neg_position = position + 1;
        eigvals.iter_mut().skip(neg_position).for_each(|v| {
            *v = zero.real();
        });
        eigvecs
            .rows_mut(neg_position, n - neg_position)
            .iter_mut()
            .for_each(|v| {
                *v = zero;
            });
    }

    // keep requested dimensionality only
    let mut eigvals = eigvals.remove_rows(number_of_dimensions, n - number_of_dimensions);
    let mut eigvecs = eigvecs.remove_rows(number_of_dimensions, n - number_of_dimensions);

    eigvals.iter_mut().for_each(|v| {
        *v = v.clone().sqrt();
    });
    mapv_inplace(&mut eigvecs, |v, row, _| {
        v * S::from_real(eigvals[(row, 0)].clone())
    });

    Some(eigvecs)
}

fn mapv_inplace<S, F>(matrix: &mut DMatrix<S>, mapv_fn: F)
where
    S: Scalar + Field + Default + Copy,
    F: Fn(S, usize, usize) -> S,
{
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            let val = matrix[(row, col)];
            matrix[(row, col)] = mapv_fn(val, row, col);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round(x: f64, decimals: u32) -> f64 {
        let y = 10i32.pow(decimals) as f64;
        (x * y).round() / y
    }

    #[test]
    fn can_center_distance_matrix_non_symmetric_f64() {
        let matrix: DMatrix<f32> = DMatrix::from_column_slice(
            3,
            3,
            &[
                0., 1., 2., //
                3., 4., 5., //
                6., 7., 8., //
            ],
        );
        let expected = vec![-3, 0, 3, 0, 0, 0, 3, 0, -3];

        let result = center_distance_matrix(matrix);
        let result = result
            .into_iter()
            .map(|&result| result.round() as i32)
            .collect::<Vec<_>>();

        assert_eq!(result, expected);
    }

    #[test]
    fn can_center_distance_matrix_symmetric_f32() {
        let matrix: DMatrix<f32> = DMatrix::from_column_slice(
            4,
            4,
            &[
                0., 7., 5., 5., //
                7., 0., 4., 9., //
                5., 4., 0., 3., //
                5., 9., 3., 0., //
            ],
        );
        let expected: Vec<f32> = vec![
            11.9375, -6.6875, -6.6875, 1.4375, //
            -6.6875, 23.6875, 3.6875, -20.6875, //
            -6.6875, 3.6875, -0.3125, 3.3125, //
            1.4375, -20.6875, 3.3125, 15.9375, //
        ];

        let result = center_distance_matrix(matrix);
        let result = result.into_iter().copied().collect::<Vec<_>>();

        assert_eq!(result, expected);
    }

    #[test]
    fn can_get_principal_coordinates() {
        let number_of_dimensions = 2;
        let eigvals =
            OVector::<f64, Dyn>::from_iterator(4, [42.1983_f64, 15.1124, 0.0000, -6.0607]);
        let eigvecs = DMatrix::from_column_slice(
            4,
            4,
            &[
                -0.2093, 0.8237, 0.5000, -0.1660, //
                0.7647, -0.0274, 0.5000, 0.4056, //
                0.05193, -0.4402, 0.5000, -0.7434, //
                -0.6073, -0.3561, 0.5000, 0.5044, //
            ],
        );
        let expected_coords = vec![
            -1.3596, 3.2021, //
            4.9675, -0.1065, //
            0.3373, -1.7113, //
            -3.9450, -1.3843, //
        ];

        let coords = get_principal_coordinates(eigvals, eigvecs, number_of_dimensions).expect("");

        let round_to = |v: &f64| round(*v, 4);
        let coords = coords.into_iter().map(round_to).collect::<Vec<_>>();

        assert_eq!(coords, expected_coords);
    }

    #[test]
    fn can_get_symmetric_eigen_decomposition() {
        let matrix: DMatrix<f64> = DMatrix::from_column_slice(
            4,
            4,
            &[
                0., 7., 5., 5., //
                7., 0., 4., 9., //
                5., 4., 0., 3., //
                5., 9., 3., 0., //
            ],
        );
        let expected_vals = vec![16.9185, -1.8377, -5.7136, -9.3672];
        let expected_vecs = vec![
            0.5039, -0.2238, -0.7961, -0.2494, //
            0.5754, 0.2955, 0.0426, 0.7614, //
            0.3775, -0.8012, 0.4643, -0.0004, //
            0.522, 0.4698, 0.3858, -0.5983, //
        ];

        let (eigvals, eigvecs) = symmetric_eigen_decomposition(matrix);

        let round_to = |v: &f64| round(*v, 4);
        let eigvals = eigvals.into_iter().map(round_to).collect::<Vec<_>>();
        let eigvecs = eigvecs.into_iter().map(round_to).collect::<Vec<_>>();
        assert_eq!(eigvals, expected_vals);
        assert_eq!(eigvecs, expected_vecs);
    }

    #[test]
    fn can_apply_pcoa() {
        let number_of_dimensions = 2;
        let matrix: DMatrix<f64> = DMatrix::from_column_slice(
            4,
            4,
            &[
                0., 7., 5., 5., //
                7., 0., 4., 9., //
                5., 4., 0., 3., //
                5., 9., 3., 0., //
            ],
        );
        let expected_coords: DMatrix<f64> = DMatrix::from_column_slice(
            4,
            2,
            &[
                1.3597, -3.2025, //
                -4.9672, 0.1067, //
                -0.3374, 1.7113, //
                3.9449, 1.3845, //
            ],
        );

        let coords = apply_pcoa(matrix, number_of_dimensions).expect("cannot apply PCoA");

        let round_to = |v: &f64| round(*v, 4);
        let coords = coords.into_iter().map(round_to).collect::<Vec<_>>();
        let expected_coords = expected_coords
            .into_iter()
            .map(round_to)
            .collect::<Vec<_>>();
        assert_eq!(coords, expected_coords);
    }
}
