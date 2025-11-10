from setuptools import setup, find_packages, Command


class PrintProjectInfoCommand(Command):
    description = 'Print project information'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # 打印项目信息
        print("Project Name:", 'Deepfake-Detection')
        print("Version:", '1.0')
        print("URL:", 'https://github.com/JackieLinn/Deepfake-Detection')
        print("License:", 'MIT License')
        print("Author:", 'JackieLinn')
        print("Author Email:", 'jackielin2024@gmail.com')
        print("Description:", 'Midterm Assessment for Introduction to Artificial Intelligence Course.')


setup(
    name='Deepfake-Detection',
    version='1.0',
    packages=find_packages(),
    url='https://github.com/JackieLinn/Deepfake-Detection',
    license='MIT License',
    author='JackieLinn',
    author_email='jackielin2024@gmail.com',
    description='Midterm Assessment for Introduction to Artificial Intelligence Course.',
    cmdclass={
        'info': PrintProjectInfoCommand,
    },
)
