
import logging

from prod.FR.frontend import AirFitApp


if __name__ == '__main__':
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info(' Starting Airfit Backend')

        app = AirFitApp()
        app.run()