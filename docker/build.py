#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Creative Commons Attribution-NonCommercial International
#   License, Version 4.0 (the "License"); you may not use this file except in
#   compliance with the License. You may obtain a copy of the License at
#
#       http://creativecommons.org/licenses/by-nc/4.0/legalcode
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
import os
import subprocess
import argparse

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # pylint: disable=W0603
ML_LEARNER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ml.Dockerfile'))  # pylint: disable=W0603
DOCKER_REGISTRY_URL = 'gcr.io/fetch-ai-sandbox'  # pylint: disable=W0603


def parse_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true', dest='publish',
                        help='Publish image to {}'.format(DOCKER_REGISTRY_URL))
    parser.add_argument('--allow_dirty', action='store_true', dest='allow_dirty', help='Allow building/pushing dirty images')
    parser.add_argument('--rebuild', action='store_true', dest='no_cache', help='Build image from scratch')
    parser.add_argument('--tag', type=str, help='Tag to use rather than git commit')
    return parser.parse_args()


def check_project_path():
    tests = [
        os.path.isdir(PROJECT_PATH),
        os.path.isdir(os.path.join(PROJECT_PATH, 'docker')),
        os.path.isfile(os.path.join(PROJECT_PATH, 'requirements.txt')),
        os.path.isfile(os.path.join(PROJECT_PATH, 'setup.py')),
    ]

    if not all(tests):
        raise RuntimeError('Failed to detect project layout')


def get_project_version():
    return subprocess.check_output(['git', 'describe', '--dirty=-dirty', '--always'], cwd=PROJECT_PATH).decode().strip()


def docker_build_multistage(dockerfile, stage, image_tag, no_cache=False, cache_from=None, publish=False):
    print('Building docker stage {} from {}...'.format(stage, dockerfile))
    cmd = [
        'docker',
        'build',
        '--ssh', 'default',
        '--target', stage,
    ]

    if cache_from is not None:
        cmd += [
            '--cache-from', cache_from,
        ]

    if no_cache is True:
        cmd += [
            '--no-cache',
        ]

    cmd += [
        '-t', image_tag,
        '-f', dockerfile,
        '.',
    ]

    print(cmd)
    subprocess.check_call(cmd, cwd=PROJECT_PATH, env=dict(os.environ, DOCKER_BUILDKIT='1'))

    # tag the image as latest
    image_tag_latest = image_tag.split(':')[0] + ':latest'
    cmd = [
        'docker',
        'tag',
        image_tag,
        image_tag_latest,
    ]
    print(cmd)
    subprocess.check_call(cmd)
    print('Building docker image {} ...complete'.format(image_tag))

    if publish:
        image_tag_remote = '{}/{}'.format(DOCKER_REGISTRY_URL, image_tag)
        cmd = [
            'docker',
            'tag',
            image_tag,
            image_tag_remote,
        ]
        print(cmd)
        subprocess.check_call(cmd)
        cmd = [
            'docker',
            'push',
            image_tag_remote
        ]
        print(cmd)
        subprocess.check_call(cmd)
        print('Publishing docker image {} ...'.format(image_tag_remote))
        print('Publishing docker image {} ...complete'.format(image_tag_remote))


def main():
    args = parse_commandline()

    # auto detect the project path
    check_project_path()

    version = get_project_version()

    if args.tag:
        version = args.tag

    docker_build_multistage(dockerfile=ML_LEARNER_PATH, stage='base', image_tag=f"ml-learner:{version}",
                            no_cache=args.no_cache, publish=args.publish)


if __name__ == '__main__':
    main()
