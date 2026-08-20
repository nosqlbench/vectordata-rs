// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! NVIDIA GPU telemetry for the resource displays.
//!
//! Absence of GPUs is the normal case, not an error: hosts without an
//! NVIDIA driver, hosts with the driver but no visible devices, and
//! builds running under a container that hides them all report "no
//! devices" and callers skip their GPU section. [`GpuMonitor::new`]
//! therefore cannot fail, and [`GpuMonitor::sample`] returns an empty
//! slice rather than an error.
//!
//! NVML is loaded at runtime (`libnvidia-ml.so` via `libloading`), so
//! this module compiles and links on machines that have never seen a
//! GPU — there is no build-time CUDA or driver dependency, and none of
//! this is gated behind the CUDA build features: monitoring a GPU host
//! is useful regardless of whether *this* binary was built to compute
//! on one.

use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::TemperatureSensor;

/// One device's telemetry at a point in time.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuSample {
    /// NVML device index (matches `nvidia-smi` ordering).
    pub index: u32,
    /// Marketing name, e.g. "NVIDIA RTX PRO 6000 Blackwell Server Edition".
    pub name: String,
    /// Percent of the sampling period during which any kernel was
    /// running (NVML's `utilization.gpu`), 0..=100.
    pub utilization_pct: u32,
    /// Percent of the sampling period during which device memory was
    /// being read or written (NVML's `utilization.memory`), 0..=100.
    /// This is bandwidth duty cycle, not the fraction of memory used.
    pub memory_io_pct: u32,
    /// Device memory in use, in bytes.
    pub memory_used_bytes: u64,
    /// Total device memory, in bytes.
    pub memory_total_bytes: u64,
    /// Core temperature in Celsius, when the driver reports one.
    pub temperature_c: Option<u32>,
    /// Instantaneous board power draw in watts, when reported.
    pub power_watts: Option<f64>,
}

impl GpuSample {
    /// Fraction of device memory in use, 0.0..=1.0 (0.0 when the device
    /// reports no memory at all, which should not happen but must not
    /// divide by zero in a render loop).
    pub fn memory_fraction(&self) -> f64 {
        if self.memory_total_bytes == 0 {
            return 0.0;
        }
        self.memory_used_bytes as f64 / self.memory_total_bytes as f64
    }

    /// A short label for narrow displays: the model without the vendor
    /// prefix and trailing edition words, e.g. "RTX PRO 6000".
    pub fn short_name(&self) -> String {
        let trimmed = self
            .name
            .trim_start_matches("NVIDIA ")
            .trim_end_matches(" Server Edition")
            .trim_end_matches(" Laptop GPU");
        trimmed.to_string()
    }
}

/// Handle to the NVML library, or `None` when this host exposes no
/// NVIDIA devices. Construct once and reuse: `Nvml::init` dlopens the
/// driver library, which is far too expensive for a render loop, while
/// [`sample`](Self::sample) is a cheap per-frame query.
pub struct GpuMonitor {
    nvml: Option<Nvml>,
    device_count: u32,
}

impl GpuMonitor {
    /// Initialize NVML. Never fails: a host without the driver, or with
    /// zero visible devices, yields a monitor that reports no devices.
    pub fn new() -> Self {
        // A driver present but reporting zero devices is treated the
        // same as no driver at all, so callers have exactly one
        // "nothing to show" condition to handle.
        match Nvml::init() {
            Ok(nvml) => {
                let device_count = nvml.device_count().unwrap_or(0);
                if device_count == 0 {
                    Self { nvml: None, device_count: 0 }
                } else {
                    Self { nvml: Some(nvml), device_count }
                }
            }
            Err(_) => Self { nvml: None, device_count: 0 },
        }
    }

    /// Whether this host has any NVIDIA device to display.
    pub fn is_available(&self) -> bool {
        self.nvml.is_some()
    }

    /// Number of visible devices (0 when unavailable).
    pub fn device_count(&self) -> u32 {
        self.device_count
    }

    /// Sample every visible device, in index order. Empty when no GPUs
    /// are present. A device that fails mid-query (driver reset, device
    /// falling off the bus) is skipped rather than failing the sweep, so
    /// a display keeps rendering its healthy peers.
    pub fn sample(&self) -> Vec<GpuSample> {
        let Some(nvml) = self.nvml.as_ref() else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(self.device_count as usize);
        for index in 0..self.device_count {
            let Ok(device) = nvml.device_by_index(index) else { continue };
            // Utilization and memory are the two required readings; a
            // device that cannot report them is not renderable.
            let Ok(util) = device.utilization_rates() else { continue };
            let Ok(mem) = device.memory_info() else { continue };
            out.push(GpuSample {
                index,
                name: device.name().unwrap_or_else(|_| format!("GPU {index}")),
                utilization_pct: util.gpu.min(100),
                memory_io_pct: util.memory.min(100),
                memory_used_bytes: mem.used,
                memory_total_bytes: mem.total,
                temperature_c: device.temperature(TemperatureSensor::Gpu).ok(),
                power_watts: device.power_usage().ok().map(|mw| mw as f64 / 1000.0),
            });
        }
        out
    }
}

impl Default for GpuMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Invariants that must hold on both GPU and GPU-less hosts — this
    /// suite runs in CI on machines with no NVIDIA driver at all.
    #[test]
    fn monitor_reports_consistently_with_or_without_devices() {
        let monitor = GpuMonitor::new();
        let samples = monitor.sample();

        if monitor.is_available() {
            assert!(monitor.device_count() > 0, "available implies devices");
            // Devices can drop out mid-sweep, so samples may be shorter
            // than the count, never longer.
            assert!(samples.len() <= monitor.device_count() as usize);
            for (pos, s) in samples.iter().enumerate() {
                assert!(s.utilization_pct <= 100, "util out of range: {s:?}");
                assert!(s.memory_io_pct <= 100, "mem io out of range: {s:?}");
                assert!(s.memory_used_bytes <= s.memory_total_bytes, "{s:?}");
                assert!((0.0..=1.0).contains(&s.memory_fraction()), "{s:?}");
                assert!(!s.name.is_empty(), "{s:?}");
                // Index order is the display contract.
                if pos > 0 {
                    assert!(s.index > samples[pos - 1].index, "unsorted: {s:?}");
                }
            }
        } else {
            assert_eq!(monitor.device_count(), 0);
            assert!(samples.is_empty(), "no devices must yield no samples");
        }
    }

    #[test]
    fn memory_fraction_handles_zero_total() {
        let s = GpuSample {
            index: 0,
            name: "test".into(),
            utilization_pct: 0,
            memory_io_pct: 0,
            memory_used_bytes: 0,
            memory_total_bytes: 0,
            temperature_c: None,
            power_watts: None,
        };
        assert_eq!(s.memory_fraction(), 0.0);
    }

    #[test]
    fn short_name_strips_vendor_and_edition() {
        let mk = |name: &str| GpuSample {
            index: 0,
            name: name.into(),
            utilization_pct: 0,
            memory_io_pct: 0,
            memory_used_bytes: 0,
            memory_total_bytes: 1,
            temperature_c: None,
            power_watts: None,
        };
        assert_eq!(
            mk("NVIDIA RTX PRO 6000 Blackwell Server Edition").short_name(),
            "RTX PRO 6000 Blackwell"
        );
        assert_eq!(mk("NVIDIA A100-SXM4-80GB").short_name(), "A100-SXM4-80GB");
        assert_eq!(mk("Tesla V100").short_name(), "Tesla V100");
    }
}
