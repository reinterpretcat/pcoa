use nalgebra::{DMatrix, Dyn, OMatrix, OVector, Scalar};
use simba::scalar::{ComplexField, Field, SupersetOf};

fn mapv_inplace<S, F>(matrix: &mut DMatrix<S>, mapv_fn: F)
where
    S: Scalar + Field + Default + Copy,
    F: Fn(S, usize, usize) -> S,
{
    let n = matrix.nrows();

    for row in 0..n {
        for col in 0..n {
            let val = matrix[(row, col)];
            matrix[(row, col)] = mapv_fn(val, row, col);
        }
    }
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

/// Calculates eigen decomposition for symmetric matrix.
/// Returns eivenvalues and eigenvectors, sorted in descending order.
fn symmetric_eigen_decomposition<S>(
    matrix: DMatrix<S>,
) -> (OVector<S::RealField, Dyn>, OMatrix<S, Dyn, Dyn>)
where
    S: Scalar + Field + SupersetOf<f64> + Default + Copy + ComplexField,
{
    let decomposition = matrix.symmetric_eigen();

    let mut eigenvals = decomposition.eigenvalues;
    let mut eigenvecs = decomposition.eigenvectors;

    // NOTE: sort eigenvalues and eigenvectors in descending order
    // TODO: this is simple, but quite inefficient way
    for _ in 0..eigenvals.len() {
        for j in 0..eigenvals.len() - 1 {
            if eigenvals[j] < eigenvals[j + 1] {
                eigenvals.swap((j, 0), (j + 1, 0));
                eigenvecs.swap_rows(j, j + 1);
            }
        }
    }

    (eigenvals, eigenvecs)
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
        0.5039, 0.5220, 0.3775, 0.5754, //
        -0.2494, -0.5983, -0.0004, 0.7614, //
        -0.7961, 0.3858, 0.4643, 0.0426, //
        -0.2238, 0.4698, -0.8012, 0.2955,
    ];

    let (eigenvalues, eigenvectors) = symmetric_eigen_decomposition(matrix);

    let round_to = |v: &f64| round(*v, 4);
    let eigenvalues = eigenvalues.into_iter().map(round_to).collect::<Vec<_>>();
    let eigenvectors = eigenvectors.into_iter().map(round_to).collect::<Vec<_>>();
    assert_eq!(eigenvalues, expected_vals);
    assert_eq!(eigenvectors, expected_vecs);
}

fn round(x: f64, decimals: u32) -> f64 {
    let y = 10i32.pow(decimals) as f64;
    (x * y).round() / y
}
