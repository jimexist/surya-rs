use candle_core::{shape::Dim, Result, Tensor};

pub trait Roll {
    fn roll<D>(&self, shift: i32, dim: D) -> Result<Self>
    where
        D: Dim + Clone,
        Self: Sized;
}

impl Roll for Tensor {
    /// Roll the tensor input along the given dimension.
    /// Elements that are shifted beyond the last position are re-introduced at the first position.
    ///
    /// ```rust
    /// # use candle_core::{Tensor, Device};
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let tensor = tensor.roll(1, 0)?;
    /// assert_eq!(tensor.to_vec2::<f32>()?, &[[4., 5.], [0., 1.], [2., 3.]]);
    /// let tensor = Tensor::new(&[[0f32, 1.], [2., 3.], [4., 5.]], &Device::Cpu)?;
    /// let tensor = tensor.roll(-1, 0)?;
    /// assert_eq!(tensor.to_vec2::<f32>()?, &[[2., 3.], [4., 5.], [0., 1.]]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    fn roll<D>(&self, shift: i32, dim: D) -> Result<Self>
    where
        D: Dim + Clone,
    {
        let dim = dim.to_index(self.shape(), "roll")?;
        let dim_size = self.dim(dim)?;
        let shift = shift.rem_euclid(dim_size as i32) as usize;
        if shift == 0 {
            Ok(self.clone())
        } else {
            let a = self.narrow(dim, 0, dim_size - shift)?;
            let b = self.narrow(dim, dim_size - shift, shift)?;
            Tensor::cat(&[&b, &a], dim)
        }
    }
}
