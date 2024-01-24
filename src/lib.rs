use nalgebra::{DMatrix, Scalar};
use simba::scalar::{Field, SupersetOf};

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
    let expected = vec![
        11.9375, -6.6875, -6.6875, 1.4375, //
        -6.6875, 23.6875, 3.6875, -20.6875, //
        -6.6875, 3.6875, -0.3125, 3.3125, //
        1.4375, -20.6875, 3.3125, 15.9375, //
    ];

    let result = center_distance_matrix(matrix);
    let result = result.into_iter().copied().collect::<Vec<_>>();

    assert_eq!(result, expected);
}
