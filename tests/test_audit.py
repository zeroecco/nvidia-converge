from nvidia_converge.audit import _parse_dpkg_packages, _parse_rpm_packages


def test_parse_dpkg_packages_filters_sorts_and_deduplicates():
    packages = _parse_dpkg_packages(
        """
zlib1g\t1.3
libnvidia-gl\t595.71.05-1ubuntu1
libnvidia-gl\t595.71.05-1ubuntu1
cuda-toolkit-13-1\t13.1.2-1
docker-ce\t5:29.4.2-2
bad line without tab
"""
    )
    assert [pkg.name for pkg in packages] == ["cuda-toolkit-13-1", "docker-ce", "libnvidia-gl"]
    assert [pkg.manager for pkg in packages] == ["apt", "apt", "apt"]
    assert all(pkg.installed for pkg in packages)


def test_parse_rpm_packages_filters_sorts_and_deduplicates():
    packages = _parse_rpm_packages(
        """
bash\t5.2-1
nvidia-container-toolkit\t1.19.0-1
nvidia-container-toolkit\t1.19.0-1
cuda-compat-13-0\t13.0.0-1
nvidia-open-595\t595.71.05-1
"""
    )
    assert [pkg.name for pkg in packages] == ["cuda-compat-13-0", "nvidia-container-toolkit", "nvidia-open-595"]
    assert [pkg.manager for pkg in packages] == ["rpm", "rpm", "rpm"]
