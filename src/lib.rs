use nalgebra::{DMatrix, Dyn, OMatrix, OVector, Scalar};
use simba::scalar::{ComplexField, Field, SupersetOf};

/// Applies Principal Coordinates Analysis on given distance (dissimilarity) matrix.
pub fn apply_pcoa<S>(distance_matrix: DMatrix<S>, number_of_dimensions: usize) -> Option<DMatrix<S>>
where
    S: Scalar + Field + SupersetOf<f64> + Default + Copy + ComplexField,
{
    // center distance matrix, a requirement for PCoA here
    let centered_matrix = center_distance_matrix(distance_matrix);

    // perform eigen decomposition
    let (eigvals, eigvecs) = symmetric_eigen_decomposition(centered_matrix);

    // get coordinates
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
/// Returns eivenvalues and eigenvectors, sorted in descending order.
fn symmetric_eigen_decomposition<S>(
    matrix: DMatrix<S>,
) -> (OVector<S::RealField, Dyn>, OMatrix<S, Dyn, Dyn>)
where
    S: Scalar + Field + SupersetOf<f64> + Default + Copy + ComplexField,
{
    let decomposition = matrix.symmetric_eigen();

    let mut eigvals = decomposition.eigenvalues;
    let mut eigvecs = decomposition.eigenvectors;

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
    // TODO assert than num_dim is less or equal matrix dim

    let n = eigvals.len();

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
            0.5039, 0.5220, 0.3775, 0.5754, //
            -0.2494, -0.5983, -0.0004, 0.7614, //
            -0.7961, 0.3858, 0.4643, 0.0426, //
            -0.2238, 0.4698, -0.8012, 0.2955,
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
                1.3597, -2.9726, //
                -5.3514, 0.1067, //
                3.248, 1.9437, //
                -1.0784, 1.5769, //
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
