# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [2022.6.0]

### Changed
- migrated to github.

## [2022.5.1]

### Fixed
- Fixes a critical bug in `AndorSDK._measure`, which makes measurement fail.
- SDK3 now follows recommended usage of cffi (`cffi.verify` is deprecated)
- better use of warning, info, error in logger

### Changed
- `AndorSDK._measure`: temporary code used to bypass old yaqd-core array protocol removed.

## [2022.5.0]

### Added
- initial release

[Unreleased]: https://github.com/yaq-project/yaqd-andor/compare/v2022.6.0...main
[2022.6.0]: https://github.com/yaq-project/yaqd-andor/compare/v2022.5.1...v2022.6.0
[2022.5.1]: https://github.com/yaq-project/yaqd-andor/compare/v2022.5.0...v2022.5.1
[2022.5.0]: https://github.com/yaq-project/yaqd-andor/releases/tag/v2022.5.0
