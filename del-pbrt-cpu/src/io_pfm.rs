use anyhow::anyhow;
use itertools::Itertools;
use std::fs::File;
use std::io::BufReader;

pub struct PFM {
    pub data: Vec<f32>,
    pub w: usize,
    pub h: usize,
    pub channels: usize,
    pub little_endian: bool,
}

impl PFM {
    pub fn read_from<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<PFM> {
        use std::io::BufRead;
        use std::io::Read;
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut header = String::new();
        reader.read_line(&mut header)?;
        let header = header.trim();

        let channels = match header {
            "PF" => 3, // color image
            "Pf" => 1, // grayscale image
            _ => return Err(anyhow!("Invalid header of PFM")),
        };

        let mut dimensions = String::new();
        reader.read_line(&mut dimensions)?;
        let dims = dimensions.split_whitespace().collect_vec();

        if dims.len() != 2 {
            return Err(anyhow!("Invalid dimensions format"));
        }

        let w: usize = dims[0]
            .parse()
            .map_err(|_| anyhow!("Invalid width: {}", dims[0]))?;
        let h: usize = dims[1]
            .parse()
            .map_err(|_| anyhow!("Invalid height: {}", dims[1]))?;

        // scale factor, little endian if scale is negative
        let mut scale_line = String::new();
        reader.read_line(&mut scale_line)?;
        let scale: f32 = scale_line
            .trim()
            .parse()
            .map_err(|_| anyhow!("Invalid scale factor: {}", scale_line.trim()))?;

        let little_endian = scale < 0.0;

        // binary data
        let data_size = w * h * channels;
        let mut buffer = vec![0u8; data_size * 4]; // 4 bytes per f32
        reader.read_exact(&mut buffer)?;

        // convert bytes to f32 values
        let mut data = Vec::with_capacity(data_size);
        for chunk in buffer.chunks_exact(4) {
            let bytes = chunk[..4].try_into().unwrap();
            let value = if little_endian {
                f32::from_le_bytes(bytes)
            } else {
                f32::from_be_bytes(bytes)
            };
            data.push(value);
        }

        Ok(PFM {
            data,
            w,
            h,
            channels,
            little_endian: true,
        })
    }
}
