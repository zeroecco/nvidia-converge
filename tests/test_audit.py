from nvidia_converge.audit import _mig_mode, _parse_dpkg_packages, _parse_rpm_packages


def test_parse_dpkg_packages_filters_sorts_and_deduplicates():
    packages = _parse_dpkg_packages(
        """
zlib1g\t1.3
libnvidia-gl\t595.71.05-1ubuntu1
libnvidia-gl\t595.71.05-1ubuntu1
cuda-toolkit-13-1\t13.1.2-1
docker-ce\t5:29.4.2-2
cuda-cub\t
bad line without tab
"""
    )
    assert [pkg.name for pkg in packages] == ["cuda-cub", "cuda-toolkit-13-1", "docker-ce", "libnvidia-gl"]
    assert [pkg.manager for pkg in packages] == ["apt", "apt", "apt", "apt"]
    assert all(pkg.installed for pkg in packages)
    assert packages[0].version is None


def test_parse_rpm_packages_filters_sorts_and_deduplicates():
    packages = _parse_rpm_packages(
        """
bash\t5.2-1
nvidia-container-toolkit\t1.19.0-1
nvidia-container-toolkit\t1.19.0-1
cuda-compat-13-0\t13.0.0-1
cuda-cub\t
nvidia-open-595\t595.71.05-1
"""
    )
    assert [pkg.name for pkg in packages] == ["cuda-compat-13-0", "cuda-cub", "nvidia-container-toolkit", "nvidia-open-595"]
    assert [pkg.manager for pkg in packages] == ["rpm", "rpm", "rpm", "rpm"]
    assert packages[1].version is None


def test_mig_mode_parses_query_output():
    assert _mig_mode("Enabled\nDisabled\n") == "enabled"
    assert _mig_mode("Disabled\nDisabled\n") == "disabled"
    assert _mig_mode("N/A\n") == "disabled"


def test_mig_mode_parses_verbose_output():
    assert _mig_mode("GPU 00000000:01:00.0\n    MIG Mode                    : Enabled\n") == "enabled"
    assert _mig_mode("GPU 00000000:01:00.0\n    MIG Mode                    : Disabled\n") == "disabled"
