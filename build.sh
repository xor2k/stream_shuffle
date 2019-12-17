rm -rf build &&
rm -rf ext &&
mkdir ext &&
touch ext/__init__.py &&
python ./setup.py build_ext --inplace