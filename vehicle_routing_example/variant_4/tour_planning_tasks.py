from os import system
from os import makedirs
from os.path import dirname, basename, abspath
import pandas as pd
import numpy as np
from os import environ
import random
import time
import requests
import json
import csv
from typing import Tuple, Dict, List, NewType, IO
from data_task import LoadDataWrapper
from os.path import join as pjoin
from cls_tasks import *
import sys
import luigi
from multimethod import multimethod
from mptop_instance_helper import *
import pickle
import types
from typing import Union

sys.path.append('../')
sys.path.append('../../')
from unique_task_pipeline_validator import UniqueTaskPipelineValidator
from inhabitation_task import ClsParameter, RepoMeta
from cls_python import FiniteCombinatoryLogic, Subtypes, Intersection, Product
from repo_visualizer.static_json_repo import StaticJSONRepo
from repo_visualizer.dynamic_json_repo import DynamicJSONRepo



GeocoordinatesDict = NewType('GeocoordinatesDict', Dict[str, Dict[str, float]])


class globalConfig(luigi.Config):
    instance_name = luigi.Parameter(default="aldi_9_1")
    load_revenue = luigi.BoolParameter(default=True)
    load_gold = luigi.BoolParameter(default=True)
    global_resources_path = luigi.Parameter(default="../resources/data")
    global_result_path = luigi.Parameter(default="results")
    global_config_path = luigi.Parameter(default="../resources/mptop_configs")
    seed = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resource_path = pjoin(
            str(self.global_resources_path), str(self.instance_name))
        self.result_path = pjoin(
            str(self.global_result_path), str(self.instance_name))
        self.result_dir_for_input_data = pjoin(
            str(self.result_path), "input_data/")
        self.scoring_result_path = pjoin(str(self.result_path), "scoring")
        self.routing_result_path = pjoin(str(self.result_path), "routing")
        self.solver_result_path = pjoin(str(self.result_path), "solver")
        self.solver_instances_result_path = pjoin(
            str(self.solver_result_path), "instances")
        self.config_result_path = pjoin(str(self.result_path), "config")
        self.hash_map_result_path = pjoin(str(self.result_path), "hash_map")
        self.best_result_path = pjoin(str(self.result_path), "best_result")


class CreateDirsTask(CLSTask, globalConfig):
    abstract = False

    def output(self):
        return luigi.LocalTarget(pjoin(self.result_path, ".dirs_created"))

    def run(self):
        makedirs(dirname(self.result_dir_for_input_data), exist_ok=True)
        makedirs(dirname(str(self.scoring_result_path) + "/"), exist_ok=True)
        makedirs(dirname(str(self.routing_result_path) + "/"), exist_ok=True)
        makedirs(dirname(str(self.solver_result_path) + "/"), exist_ok=True)
        makedirs(dirname(str(self.solver_instances_result_path) + "/"), exist_ok=True)
        makedirs(dirname(str(self.config_result_path) + "/"), exist_ok=True)
        makedirs(dirname(str(self.hash_map_result_path) + "/"), exist_ok=True)
        makedirs(dirname(str(self.best_result_path) + "/"), exist_ok=True)
        with open(self.output().path, 'w') as file:
            pass


class AbstractGatherAndIntegratePhase(CLSTask, globalConfig):
    abstract = True
    input_data = ClsParameter(tpe=LoadDataWrapper.return_type())

    def requires(self):
        return {"create_dir": CreateDirsTask(), "gather_and_integrate": self.input_data(self.instance_name, self.load_revenue, self.load_gold, self.resource_path, self.result_dir_for_input_data)}

    def output(self):
        return self.input()["gather_and_integrate"]

    def run(self):
        pass


class AbstractRoutingPhase(CLSTask, globalConfig):
    abstract = True
    gather_and_integrate_phase = ClsParameter(
        tpe=AbstractGatherAndIntegratePhase.return_type())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lat_lon_results: Dict[int, Tuple[float, float]] = {}
        self.routing_results: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def requires(self):
        return self.gather_and_integrate_phase()

    def output(self):
        return {"lat_lon_result": luigi.LocalTarget(pjoin(self.routing_result_path, self._get_variant_label() + "-" + "lat_lon_result.csv")),
                "dima_result": luigi.LocalTarget(pjoin(self.routing_result_path, self._get_variant_label() + "-" + "dima_result.dima")),
                "csv_dima_result": luigi.LocalTarget(pjoin(self.routing_result_path, self._get_variant_label() + "-" + "dima_result.csv"))
                }

    def run(self):
        self._routing_method()
        self._create_result_lat_lon()
        self._create_result_dima()

    def _routing_method(self):
        """
        This method represents the routing method. 
        -----------------------------------------------------------------------

        There are no real restrictions or guidelines on how to implement your routing method. You only need to fill
        the to lists *lon_lat_results* and *routing_results*. For lon_lat_results you need to use the methods
        "add_lon_lat_result()" for the sales_person and all customers. This information will be used to create the
        MPTOP Instance in a later stage. For the routing_results list you need to use the methods "add_dima_result()"
        and add a result for each sales_person - customer / customer - customer combination.
        """
        raise NotImplementedError("Has to be implemented by concrete classes")

    def _create_result_lat_lon(self) -> None:
        """
        This method creates the lon lat result file that is used to create the mptop instance file.

        Uses the "lat_lon_results" dictionary so you need to make sure it is filled.
        :return: None
        """
        content = "id,lat,lon\n"
        for i, (id, (lat, lon)) in enumerate(self._get_lat_lon_results().items()):
            content += str(id) + "," + str(lat) + "," + str(lon) + "\n" if i < len(self._get_lat_lon_results().items()) \
                - 1 else str(id) + "," + str(lat) + "," \
                + str(lon) + ""

        with open(self.output()["lat_lon_result"].path, "w") as file:
            file.write(content)

    def _create_result_dima(self) -> None:
        """
        This method creates the dima result file that is used for the mptop algorithm.

        Uses the "routing_results" dictionary so you need to make sure it is filled.
        :return: None
        """
        content = "Distance_Matrix\n"
        content += "from;to;distance;time\n"
        for i, ((from_id, to_id), (distance, time)) in enumerate(self._get_routing_results().items()):
            content += str(from_id) + ";" + str(to_id) + ";" + str(distance) + ";" + str(time) \
                + "\n" if i < len(self._get_routing_results().items()) - 1 else str(from_id) + ";" + str(to_id) \
                + ";" + str(distance) \
                + ";" + \
                str(time) + ""
        with open(self.output()["dima_result"].path, "w") as file:
            file.write(content)
        csv_content = ""
        lines = open(self.output()["dima_result"].path, "r").readlines()[1:]
        for line in lines:
            csv_content += line.replace(";", ",")
        with open(self.output()["csv_dima_result"].path, "w") as file:
            file.write(csv_content)

    @multimethod
    def _add_lat_lon_result(self, id: int, lat: float, lon: float) -> None:
        """
        This method adds a line into the lon lat result.

        :param id: id of either a customer or a sales person.
        :param lat: lat position of that person.
        :param lon: lon position of that person.
        :return: None
        """
        self.lat_lon_results[id] = (lat, lon)

    @multimethod
    def _add_lat_lon_result(self, result_tuple: Tuple[int, float, float]) -> None:
        """
        This method adds a line into the lon lat result.
        :param result_tuple: (Tuple[int,float,float])

        :return: None
        """
        self.lat_lon_results[result_tuple[0]] = (
            result_tuple[1], result_tuple[2])

    @multimethod
    def _add_lat_lon_result(self, list_of_results: Tuple[int, float, float]) -> None:
        """
        This method adds a line for every item in the list into the lon lat result.
        :param list_of_results: (List[Tuple[int,float,float]]): List[Tuple[id, lat, lon]]

        :return: None
        """
        element: Tuple[int, float, float]
        for tpl in list_of_results:
            self._add_lat_lon_result(tpl)

    def _get_lat_lon_results(self) -> Dict[int, Tuple[float, float]]:
        """
        Returns the routing results list.

        :return: List[Tuple[int, float, float]]
        """
        return self.lat_lon_results

    @multimethod
    def _add_dima_result(self, from_id: int, to_id: int, distance: int, time: int) -> None:
        """
        This method adds a line into the distance/time result matrix.
        Every result line in the DIMA has the following form: "from;to;distance;time"
        :param from_id: customer id that is the starting point.
        :param to_id: customer id that we want to reach.
        :param distance: distance between "from_id" to "to_id" in meters.
        :param time: travel time between "from_id" to "to_id" in seconds.

        :return: None
        """
        self.routing_results[(from_id, to_id)] = (distance, time)

    @multimethod
    def _add_dima_result(self, result_tuple: Tuple[int, int, int, int]) -> None:
        """
        This method adds a line into the distance/time result matrix.
        Every result line in the DIMA has the following form: "from;to;distance;time"
        :param result_tuple: (Tuple[int,int,int,int]): Tuple[from_id, to_id, distance, time]

        :return: None
        """
        self.routing_results[(result_tuple[0], result_tuple[1])] = (
            result_tuple[2], result_tuple[3])

    @multimethod
    def _add_dima_result(self, list_of_results: List[Tuple[int, int, int, int]]) -> None:
        """
        This method adds a line for every item in the list into the distance/time result matrix.
        Every result line in the DIMA has the following form: "from;to;distance;time"
        :param list_of_results: (List[Tuple[int,int,int,int]]): List[Tuple[from_id, to_id, distance, time]]

        :return: None
        """
        for tpl in list_of_results:
            self._add_dima_result(tpl)

    def _get_routing_results(self) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """
        Returns the routing results dictionary.

        :return: Dict[Tuple[int, int], Tuple[int, int]]
        """
        return self.routing_results


class AbstractScoringPhase(CLSTask, globalConfig):
    abstract = True
    gather_and_integrate_phase = ClsParameter(
        tpe=AbstractGatherAndIntegratePhase.return_type())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scoring_results: Dict[int, Tuple[float, int]] = {}

    def requires(self):
        return self.gather_and_integrate_phase()

    def output(self):
        return {"scoring_result": luigi.LocalTarget(pjoin(self.scoring_result_path, self._get_variant_label() + "-" + "scoring_result.csv")),
                "scoring_method": luigi.LocalTarget(pjoin(self.scoring_result_path, "." + self._get_variant_label() + "-scoring_method"))}

    def run(self):
        self._scoring_method()
        self._create_result_csv()
        self._create_which_config_file()

    def _scoring_method(self):
        """
        This method represents the scoring method, which needs to be implemented by concrete classes.

        There are no real restrictions or guidelines on how to implement your scoring method. The only thing that the
        pipeline needs you to do is fill the scoring_results dictionary with the calculated scoring values for each
        customer in the form of id -> (value, abcCategory). To do so, you need to use the "add_result()" method and
        add a result for each customer.

        There is a "add_result()" method:

        1. add_result(id: str, value: float, abc_category: int = 0) : Adds a result for a customer with the given id.
        Will set the abcCategory to 0 by default. Example: add_result(100,3.33)

        2. The method can also be used to set different abc_category values (0-2). Example add_result(100,3.33,1)

        :return: None
        """
        raise NotImplementedError("Has to be implemented by concrete classes")

    def _create_result_csv(self) -> None:
        """
        This method creates the csv result file that is used to create the mptop instance.
        Uses the "scoring_results" dictionary so you need to make sure it is filled.

        :return: None
        """
        content = "customer_id,value,abc\n"
        for i, (customer_id, (value, abc)) in enumerate(self._get_scoring_results().items()):
            content += str(customer_id) + "," + str(value) + "," + str(abc) \
                + "\n" if i < len(self._get_scoring_results().items()) - 1 else str(customer_id) \
                + "," + str(value) + "," \
                + str(abc) + ""
        with open(self.output()["scoring_result"].path, "w") as file:
            file.write(content)

    def _get_scoring_results(self) -> Dict[int, Tuple[float, int]]:
        """
        This method can be used to get the scoring results dictionary.

        :return: the dictionary of scoring results.
        """
        return self.scoring_results

    def _add_result(self, id: int, value: float, abc_category: int = 0) -> None:
        """
        This method adds a scoring value for a customer to the overall result.

        The result should be a list of tuples  with the customer id, the calculated customer score as value,
        and the abcCategory. The result will be used to create a csv file with the content, which is later used to
        create the mptop instance.

        :param id: should be the id of the customer. Comes from the database and is associated
                    with the id/extId within the database.
        :param value: scoring value
        :param abc_category: A=2, B=1, C=0, Default is 0, if you use it without a definition of a category.

        :return: None
        """
        self.scoring_results[id] = (value, abc_category)

    def _create_which_config_file(self):
        with open(self.output()["scoring_method"].path, "w") as file:
            file.write(self._get_scoring_method_string())


class GatherAndIntegratePhase(AbstractGatherAndIntegratePhase, globalConfig):
    abstract = False


class OsrmRoutingPhase(AbstractRoutingPhase):
    abstract = False

    def _routing_method(self) -> None:
        """
        This method represents the routing method which uses osrm to calculate a dima. To calculate lat and lon values
        a nominatim Server is used.

        :return: None
        """
        sales_person = open(self.input()["sales_person"].path, "r")
        geocoordiantes_sales_person = self._get_list_of_geocoordiantes(
            sales_person)
        print("Sales PERSON: ")
        print(geocoordiantes_sales_person)

        customers = open(self.input()["customers"].path, "r")
        geocoordiantes_customers = self._get_list_of_geocoordiantes(customers)
        print("Customer: ")
        print(geocoordiantes_customers)

        sales_person.close()
        customers.close()

        # Create dict
        nodes_for_dima_calc = self._create_dict_for_osrm_call(
            geocoordiantes_sales_person, geocoordiantes_customers)

        osrm_query = self._get_osrm_backend_query(nodes_for_dima_calc)

        print("OSRM Query:")
        print(osrm_query)

        osrm_response = requests.get(osrm_query, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/51.0.2704.103 Safari/537.36"})

        print("OSRM Response:")
        print(osrm_response)

        if osrm_response.status_code == 503:
            print("OSRM service is not available! Execution stopped.")
            sys.exit()
        elif osrm_response.status_code == 200:
            osrm_response_json = json.loads(osrm_response.text)
            if osrm_response_json['code'] != "Ok":
                print("Something went wrong! Execution stopped.")
                print(osrm_response_json)
                sys.exit()
            else:
                # Call was successful and we now need to create the dima file out of it.
                for i, _ in enumerate(nodes_for_dima_calc):

                    distances = osrm_response_json['distances'][i]
                    durations = osrm_response_json['durations'][i]
                    node_i_id = nodes_for_dima_calc[i]['id']
                    for j, _ in enumerate(nodes_for_dima_calc):
                        distance_i_to_j = str(distances[j])
                        duration_i_to_j = str(durations[j])
                        node_j_id = nodes_for_dima_calc[j]['id']
                        self._add_dima_result(int(node_i_id), int(node_j_id), int(round(float(distance_i_to_j))),
                                              int(round(float(duration_i_to_j))))

        else:
            print("Something went wrong! Execution stopped.")
            sys.exit()

    def _get_list_of_geocoordiantes(self, table_csv: IO) -> GeocoordinatesDict:
        """
        This method translates a location to longitude and latitude.
        """
        result_dict: GeocoordinatesDict = GeocoordinatesDict(dict())
        csv_reader = csv.DictReader(table_csv)
        line_count = 0
        for person in csv_reader:
            print(person)
            try:
                id = person["customer_id"].strip().replace(" ", '%20')
            except KeyError:
                id = person["sales_person_id"].strip().replace(" ", '%20')
            city = person["city"].strip().replace(" ", '%20')
            street = person["street"].strip().replace(" ", '%20')
            house_number = person["house_number"].strip().replace(" ", '%20')
            postal_code = person["postal_code"].strip().replace(" ", '%20')
            country = person["country"].strip().replace(" ", '%20')

            line_count += 1

            nominatim_query = "http://localhost:8080/search?street=" + street + "%20" + house_number + "&city=" + city \
                + "&country=" + country + "&postalcode=" + postal_code + "&polygon_geojson=1&format" \
                "=jsonv2"
            print("Nominatim Query:")
            print(nominatim_query)
            nominatim_response = requests.get(nominatim_query,
                                              headers={"User-Agent": "Mozilla/5.0 ("
                                                                     "X11; Linux "
                                                       "x86_64) AppleWebKit/537.36 ("
                                                       "KHTML, like Gecko) "
                                                       "Chrome/51.0.2704.103 "
                                                       "Safari/537.36"})
            print("Nominatim Response:")
            print(nominatim_response)
            while not nominatim_response.status_code == 200:
                time.sleep(30.0)
                print("sleept")
                nominatim_response = requests.get(nominatim_query,
                                                  headers={"User-Agent": "Mozilla/5"
                                                                         ".0 (X11; "
                                                           "Linux "
                                                           "x86_64) "
                                                           "AppleWebKit/537.36 "
                                                           " (KHTML, like Gecko) "
                                                           "Chrome/51.0.2704.103 "
                                                           "Safari/537.36"})
                print("Nominatim Response:")
                print(nominatim_response)

                if nominatim_response.status_code == 503:
                    print("Nominatim service is not available! Execution stopped.")
                    sys.exit()

            nominatim_response_json = json.loads(nominatim_response.text)

            lat = nominatim_response_json[0]["lat"]
            lon = nominatim_response_json[0]["lon"]
            result_dict[id] = {'lat': lat, 'lon': lon}
            print(
                f'Processed {line_count} customers and calculated lat lon values.')

            # Add lon lat Coordinates to lon_lat_results
            self._add_lat_lon_result(int(id), float(lat), float(lon))

        return result_dict

    def _create_dict_for_osrm_call(self, geocoordiantes_sales_person, geocoordiantes_customers):
        """
        This method create a dict we can use to call the osrm_backend and get the dima
        """

        nodes_for_dima_calc = {}

        # TODO: Mögliche Verbesserung, auch eine for Schleife für alle Sales-Persons, anstatt nur eine Person.
        (id, values) = list(geocoordiantes_sales_person.items())[0]

        nodes_for_dima_calc[0] = {

            "id": id,
            "lat": values['lat'],
            "lon": values['lon']
        }

        i = 1
        for id, values in geocoordiantes_customers.items():
            nodes_for_dima_calc[i] = {

                "id": id,
                "lat": values['lat'],
                "lon": values['lon']

            }
            i = i + 1
        return nodes_for_dima_calc

    def _get_osrm_backend_query(self, nodes_for_dima_calc):
        """
        This method gets a dict in a specific format and returns the osrm backend call we need to perform
        in order to get the dima.
        """

        # lon lat
        node_count = len(nodes_for_dima_calc)
        osrm_query = ""
        i = 0
        for node in nodes_for_dima_calc:
            osrm_query = osrm_query + \
                nodes_for_dima_calc[node]['lon'] + "," + \
                nodes_for_dima_calc[node]['lat']

            if i < node_count - 1:
                osrm_query = osrm_query + ";"
            i = i + 1

        return "http://localhost:5000/" + "table/v1/driving/" + osrm_query + "?annotations=distance,duration"


class DistanceMatrixAiRoutingPhase(AbstractRoutingPhase):
    """
    Implementation of a RoutingPhase that uses the DistanceMatrixAi Webservice. Make sure to set the 
    environment variable DISTANCEMATRIXAIAPI as a environment variable and provide your personal API Key.
    """
    abstract = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = environ['DISTANCEMATRIXAIAPI']

    def _routing_method(self) -> None:
        """
        This method represents the routing method which uses a webservice to calculate a DIMA and Lat_Lon values.
        The webservice used is DistanceMatrixAi.
        :return: None
        """
        # Create latlon Results.
        customers = open(self.input()["customers"].path, "r")
        csv_reader = csv.DictReader(customers)
        line_count = 0
        for row in csv_reader:
            self._do_geocall_and_fill_latlon_results(row)
            line_count += 1
        print(
            f'Processed {line_count} customers and calculated lat and lon values.')
        customers.close()

        sales_person = open(self.input()["sales_person"].path, "r")
        csv_reader = csv.DictReader(sales_person)
        line_count = 0
        for row in csv_reader:
            self._do_geocall_and_fill_latlon_results(row)
            line_count += 1
        print(
            f'Processed {line_count} sales persons and calculated lat and lon values.')
        sales_person.close()

        # Create DIMA Results.

        print("================================================")
        for i_id, (i_lat, i_lon) in self._get_lat_lon_results().items():
            for j_id, (j_lat, j_lon) in self._get_lat_lon_results().items():
                if i_id == j_id:
                    self._add_dima_result(i_id, j_id, 0, 0)
                else:
                    destination_id_latlon_list = []
                    destination_latlon_list = []
                    destination_id_latlon_list.append((j_id, j_lat, j_lon))
                    destination_latlon_list.append((j_lat, j_lon))
                    self._do_distance_matrix_call_and_fill_dima_results(i_id, destination_id_latlon_list,
                                                                        self._get_distance_matrix_call(
                                                                            self._get_distance_matrix_parameter((i_lat, i_lon), destination_latlon_list)))
            print("Filled the dima result for id: " + str(i_id))

    def _do_geocall_and_fill_latlon_results(self, row) -> None:
        """
        This method does the geocoding api call and adds the result to the latlon results list.
        :param row: of a customer or sales person
        :return: None
        """
        geoservice_response = requests.get(
            self._get_geocoding_call(self._get_address_parameter(row)))
        print(self._get_geocoding_call(self._get_address_parameter(row)))
        if geoservice_response.status_code == 503:
            print("Service is not available! Execution stopped.")
            sys.exit()
        while geoservice_response.status_code != 200:
            time.sleep(2)
            geoservice_response = requests.get(
                self._get_geocoding_call(self._get_address_parameter(row)))

        geoservice_response_json = json.loads(geoservice_response.text)
        if geoservice_response_json['status'] != "OK":
            print(geoservice_response_json['status'])
            print(geoservice_response_json)
            print("Something went wrong! Execution stopped.")
            sys.exit()
        else:
            lat = geoservice_response_json["result"][0]["geometry"]["location"]["lat"]
            lon = geoservice_response_json["result"][0]["geometry"]["location"]["lng"]
            try:
                self._add_lat_lon_result(
                    int(row["customer_id"].strip()), float(lat), float(lon))
            except KeyError:
                self._add_lat_lon_result(
                    int(row["sales_person_id"].strip()), float(lat), float(lon))

    def _get_geocoding_call(self, parameter: str) -> str:
        """
        This method returns the URL for the Geocoding API call.
        :param parameter: should be the result of the getAdressParameter() method.
        :return: str: the URL
        """
        return "https://api.distancematrix.ai/maps/api/geocode/json?" + parameter + "&language=de&key" \
                                                                                    "=" + self.api_key + ""

    def _get_address_parameter(self, row) -> str:
        """
        This method creates the parameter string for the geocoding api call.
        :param row: of a customer or sales person
        :return: the string that represents the parameter part of the api call.
        """
        street_name = row["street"].strip().replace(" ", "+")
        street_number = row["house_number"].strip().replace(" ", "+")
        zipcode = row["postal_code"].strip().replace(" ", "+")
        city = row["city"].strip().replace(" ", "+")
        country = row["country"].strip().replace(" ", "+")
        delimiter = "%20"
        return "address=" + street_name + delimiter + street_number + delimiter + zipcode + delimiter + city \
               + delimiter \
               + country + ""

    def _get_distance_matrix_call(self, parameter: str) -> str:
        """
        This method returns the URL for the DIMA API call.
        :param parameter:
        :return:
        """
        return "https://api.distancematrix.ai/maps/api/distancematrix/json?" + parameter + "&language=de&key=" + self.api_key + ""

    def _get_distance_matrix_parameter(self, origin: Tuple[float, float], destination: List[Tuple[float, float]]) -> str:
        """
        This method returns the Parameterlist for the DIMA API Call URL.
        :param origin: lat and lon as a Tuple
        :param destination: A list of lat and long of Tuples
        :return: parameterlist as a string
        """
        dest_temp = ""
        for i, (lat, lon) in enumerate(destination):
            dest_temp = dest_temp + \
                str(lat) + "," + str(lon) + \
                ("|" if i < len(destination) - 1 else "")

        return "origins=" + str(origin[0]) + "," + str(origin[1]) + "&destinations=" + dest_temp

    def _do_distance_matrix_call_and_fill_dima_results(self, from_id, to_list_id_latlon, api_call: str) -> None:
        """
        This method is doing the call to the distance matrix api and fills the results into the results list.
        :param from_id: the id of the starting point
        :param to_list_id_latlon: a list of id + latlon values. Are used to calculate the distance/duration between
               from_id and the list.
        :param api_call: the URL for the api call. Is used to request the results.
        :return: None
        """

        distance_matrix_response = requests.get(api_call)
        print(api_call)
        if distance_matrix_response.status_code == 503:
            print("Service is not available! Execution stopped.")
            sys.exit()
        while distance_matrix_response.status_code != 200:
            time.sleep(2)
            distance_matrix_response = requests.get(api_call)

        distance_matrix_response_json = json.loads(
            distance_matrix_response.text)
        if distance_matrix_response_json['status'] != "OK":
            print(distance_matrix_response_json['status'])
            print("Something went wrong! Execution stopped.")
            sys.exit()
        else:
            print(distance_matrix_response_json["rows"][0]["elements"])
            for i, element in enumerate(distance_matrix_response_json["rows"][0]["elements"]):
                if element["status"] != "OK":
                    print("Error in Element:")
                    print(element)
                    # TODO: STUFF
                    print("Service is not able to calculate distance/time between id:" +
                          str(from_id)+" and id:"+str(to_list_id_latlon[i][0])+"!")
                    continue
                    # print("Execution stopped")
                    # sys.exit()
                else:
                    distance = int(element["distance"]["value"])
                    duration = int(element["duration"]["value"])
                    to_id = int(to_list_id_latlon[i][0])
                    self._add_dima_result(
                        int(from_id), to_id, distance, duration)


class SabcScoringPhase(AbstractScoringPhase):
    abstract = False

    def _get_scoring_method_string(self):
        return "sabc"

    def _scoring_method(self) -> None:
        """
        This is the strict ABC Scoring Method. The Way it calculates its scouring's can be found in

        @article{meyer2020planning,
          title={Planning profitable tours for field sales forces: A unified view on sales analytics and mathematical optimization},
          author={Meyer, Anne and Glock, Katharina and Radaschewski, Frank},
          journal={arXiv preprint arXiv:2011.14822},
          year={2020}
        }

        :return: None
        """
        sales = self._get_csv_as_pandas_dataframe(
            self.input()["customers_revenue"].path)
        sales.sort_values(by=['revenue'], ascending=False, inplace=True)
        total = sales['revenue'].sum()

        revenue_in_percent = []
        accumulated_revenue = []

        for column in sales:
            if sales[column].name == "revenue":
                accumulated = 0.0
                for item in sales[column]:
                    revenue_in_percent.append(item / total)
                    accumulated += item / total
                    accumulated_revenue.append(accumulated)
                # print(sales[column])

        sales['revenue_in_percent'] = revenue_in_percent
        sales['accumulated_revenue'] = accumulated_revenue

        sales['ABC'] = np.where((sales['accumulated_revenue'] < 0.75) | (sales['revenue_in_percent'] >= 0.75), 'A',
                                np.where(sales['accumulated_revenue'] <= 0.95, 'B', 'C'))
        abc_count = sales['ABC'].value_counts(sort=True)

        p_c: float = 1.0
        p_b: float = p_c * abc_count['C'] + 1
        p_a: float = p_c * abc_count['C'] + p_b * abc_count['B'] + 1

        for item in sales.itertuples(index=False):
            if item.ABC == 'C':
                value = p_c
                abc = 0
            elif item.ABC == 'B':
                value = p_b
                abc = 1
            else:
                value = p_a
                abc = 2
            self._add_result(int(item.customer_id), value, abc)
        print("=$=$=$=$=$$==$=$=$=$$$$$$$$$$$$$$$$$$$$$$$=$==$=$=$=$=$")
        print(self._get_scoring_results())

    def _get_csv_as_pandas_dataframe(self, path_to_file: str) -> pd.DataFrame():
        """
        This function returns a csv file as a pandas dataframe.

        :param path_to_file:  full path of the file you want to load as pandas dataframe.
        :return: pandas DataFrame
        """
        df = pd.read_csv(path_to_file)
        return df


class WabcScoringPhase(AbstractScoringPhase):
    abstract = False

    def _get_scoring_method_string(self):
        return "wabc"

    def _scoring_method(self) -> None:
        """
        This is the weighted ABC Scoring Method. The Way it calculates its scouring's can be found in

        @article{meyer2020planning,
          title={Planning profitable tours for field sales forces: A unified view on sales analytics and mathematical optimization},
          author={Meyer, Anne and Glock, Katharina and Radaschewski, Frank},
          journal={arXiv preprint arXiv:2011.14822},
          year={2020}
        }

        :return: None
        """
        sales = self._get_csv_as_pandas_dataframe(
            self.input()["customers_revenue"].path)
        sales.sort_values(by=['revenue'], ascending=False, inplace=True)
        total = sales['revenue'].sum()

        revenue_in_percent = []
        accumulated_revenue = []

        for column in sales:
            if sales[column].name == "revenue":
                accumulated = 0.0
                for item in sales[column]:
                    revenue_in_percent.append(item / total)
                    accumulated += item / total
                    accumulated_revenue.append(accumulated)
                # print(sales[column])

        sales['revenue_in_percent'] = revenue_in_percent
        sales['accumulated_revenue'] = accumulated_revenue

        sales['ABC'] = np.where((sales['accumulated_revenue'] < 0.75) | (sales['revenue_in_percent'] >= 0.75), 'A',
                                np.where(sales['accumulated_revenue'] <= 0.95, 'B', 'C'))

        p_c: float = 1.0
        p_b: float = p_c * 5
        p_a: float = p_b * 3

        for item in sales.itertuples(index=False):
            if item.ABC == 'C':
                value = p_c
                abc = 0
            elif item.ABC == 'B':
                value = p_b
                abc = 1
            else:
                value = p_a
                abc = 2
            self._add_result(int(item.customer_id), value, abc)

        print(self._get_scoring_results())

    def _get_csv_as_pandas_dataframe(self, path_to_file: str) -> pd.DataFrame():
        """
        This function returns a csv file as a pandas dataframe.

        :param path_to_file:  full path of the file you want to load as pandas dataframe.
        :return: pandas DataFrame
        """
        df = pd.read_csv(path_to_file)
        return df


class NsScoringPhase(AbstractScoringPhase):
    abstract = False

    def _get_scoring_method_string(self):
        return "ns"

    def _scoring_method(self) -> None:
        """
        This is the NS Scoring Method.

        :return: None
        """
        customers = open(self.input()["customers"].path, "r")
        csv_reader = csv.DictReader(customers)
        line_count = 0
        for customer in csv_reader:
            customer_id = customer["customer_id"].strip()
            self._add_result(int(customer_id), 1.0, 2)
            line_count += 1
        customers.close()


class RandomScoringPhase(AbstractScoringPhase):
    abstract = False

    def _get_scoring_method_string(self):
        return ""

    def _scoring_method(self) -> None:
        """
        This is the random scoring method. Using this method customers just get an abc value of B with a random value
        between 0.0 and 100.0. If a customer is a goldmember we assign A as the abc value with a random value
        between 0.0 and 100.0.

        :return: None
        """

        customers = open(self.input()["customers"].path, "r")
        gold = open(self.input()["goldmember"].path, "r")
        customer_csv_reader = csv.DictReader(customers)
        gold_csv_reader = csv.DictReader(gold)
        line_count = 0
        for customer in customer_csv_reader:
            customer_id = customer["customer_id"].strip()
            is_gold = False
            for goldmember in gold_csv_reader:
                gold_id = goldmember["customer_id"].strip()
                if customer_id == gold_id:
                    is_gold = True if goldmember["is_goldmember"].strip(
                    ) == "1" else False
                    break
            value = round(random.uniform(0.0, 100.0), 1)
            if is_gold:
                value = value + 50.0
            self._add_result(int(customer_id), value, 2 if is_gold else 1)
            line_count += 1
        print(
            f'Processed {line_count} customers and calculated random scoring values.')
        customers.close()
        gold.close()
        print(self._get_scoring_results())


class AbstractMptopConfig(CLSTask, globalConfig):
    abstract: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def output(self):
        return luigi.LocalTarget(pjoin(self.config_result_path, self._get_variant_label() + "-" + "mptop_config.yaml"))

    def run(self):
        with open(self.config_path, "r") as source_file, open(self.output().path, "w") as file:
            for line in source_file:
                file.write(line)

class AbstractNSConfig(AbstractMptopConfig):
    abstract = True
    ns_scoring = ClsParameter(tpe=NsScoringPhase.return_type())

    def requires(self):
        return self.ns_scoring()
    
class AbstractSABCConfig(AbstractMptopConfig):
    abstract = True
    sabc_scoring = ClsParameter(tpe=SabcScoringPhase.return_type())

    def requires(self):
        return self.sabc_scoring()
    
class AbstractWABCConfig(AbstractMptopConfig):
    abstract = True
    wabc_scoring = ClsParameter(tpe=WabcScoringPhase.return_type())

    def requires(self):
        return self.wabc_scoring()



# class AbstractNSConfig(AbstractMptopConfig):
#     abstract = True
#     ns_scoring = ClsParameter(tpe={1: NsScoringPhase.return_type(), 2: RandomScoringPhase.return_type()})
    
#     config_index = {1,2}
#     def requires(self):
#         return self.ns_scoring()
    

# class AbstractSABCConfig(AbstractMptopConfig):
#     abstract = True
#     sabc_scoring = ClsParameter(tpe={1: SabcScoringPhase.return_type(), 2: RandomScoringPhase.return_type()})
    
#     config_index = {1,2}
#     def requires(self):
#         return self.sabc_scoring()
    
# class AbstractWABCConfig(AbstractMptopConfig):
#     abstract = True
#     wabc_scoring = ClsParameter(tpe={1: WabcScoringPhase.return_type(), 2: RandomScoringPhase.return_type()})
    
#     config_index = {1,2}
#     def requires(self):
#         return self.wabc_scoring()

# class AbstractRandomConfig(AbstractMptopConfig):
#     abstract = True
#     random_scoring = ClsParameter(tpe= RandomScoringPhase.return_type())
    
#     def requires(self):
#         return self.random_scoring()
    
#     def new_requires(self):
#         return RandomScoringPhase()
    
    
    
class NSConfig1(AbstractNSConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(
            self.global_config_path, "benchmark_ns/NS_config_1.yaml")


class NSConfig2(AbstractNSConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(
            self.global_config_path, "benchmark_ns/NS_config_2.yaml")


class NSConfig3(AbstractNSConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(
            self.global_config_path, "benchmark_ns/NS_config_3.yaml")


class SABCConfig1(AbstractSABCConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(
            self.global_config_path, "benchmark_sabc/sABC_config_1.yaml")


class SABCConfig2(AbstractSABCConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(
            self.global_config_path, "benchmark_sabc/sABC_config_2.yaml")


class SABCConfig3(AbstractSABCConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(
            self.global_config_path, "benchmark_sabc/sABC_config_3.yaml")


class WABCConfig1(AbstractWABCConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(
            self.global_config_path, "benchmark_wabc/wABC_config_1.yaml")


class WABCConfig2(AbstractWABCConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(
            self.global_config_path, "benchmark_wabc/wABC_config_2.yaml")


class WABCConfig3(AbstractWABCConfig):
    abstract: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = pjoin(
            self.global_config_path, "benchmark_wabc/wABC_config_3.yaml")


# class AllConfigWrapper(AbstractRandomConfig):
#     abstract = False
#     config = ClsParameter(tpe={1: NSConfig1.return_type(), 2: NSConfig2.return_type()})
    
#     def requires(self):
#         config_task = self.config()
#         print(dir(config_task))
#         setattr(config_task, "requires", self.new_requires)
#         return config_task

class AbstractSolverPhase(CLSTask, globalConfig):
    abstract = True
    gather_and_integrate_phase = ClsParameter(
        tpe=AbstractGatherAndIntegratePhase.return_type())
    scoring_phase = ClsParameter(tpe=AbstractScoringPhase.return_type())
    routing_phase = ClsParameter(tpe=AbstractRoutingPhase.return_type())
    config = ClsParameter(tpe=AbstractMptopConfig.return_type())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def requires(self):
        return {"gather_and_integrate_phase": self.gather_and_integrate_phase(), "scoring_phase": self.scoring_phase(), "routing_phase": self.routing_phase(), "config": self.config()}

    def output(self):
        return {"solver_result": luigi.LocalTarget(pjoin(str(self.solver_result_path), self._get_variant_label() + "-" + "solver_result.txt"))}

    def run(self):
        instance_file_path: str = self._create_solver_instance()
        solver_result_file_path: str = self._run_solver(instance_file_path)
        self._create_result_file(solver_result_file_path)

    def _create_result_file(self, solver_result_file_path: str):
        """
        Takes the result file of the _run_solver() method and creates the luigi.LocalTarget. 
        """
        final_result_content = ""
        with open(solver_result_file_path, "r") as source_file:
            for line in source_file:
                final_result_content += line

        with open(self.output()["solver_result"].path, "w") as result_file:
            result_file.write(final_result_content)


class MptopSolver(AbstractSolverPhase):
    abstract = False
    seed = luigi.IntParameter(default=1)

    def output(self):
        parent_output = super().output()
        parent_output["mptop_log"] = luigi.LocalTarget(pjoin(
            str(self.solver_result_path), self._get_variant_label() + "-" + "mptop_log.txt"))
        return parent_output

    def _create_solver_instance(self):
        customers_dict = self._get_customers_result_dict()
        sales_person_dict = self._get_sales_person_result_dict()
        return self._create_mptop_instance(customers_dict, sales_person_dict)

    def _get_customers_result_dict(self):
        result = {}
        with open(self.input()["scoring_phase"]["scoring_result"].path, "r") as scoring_results, \
                open(self.input()["routing_phase"]["lat_lon_result"].path, "r") as latlon_results, \
                open(self.input()["gather_and_integrate_phase"]["customers"].path, "r") as customers:

            csv_reader_scoring_results = csv.DictReader(scoring_results)

            for scoring_line in csv_reader_scoring_results:
                sid = scoring_line["customer_id"].strip()
                latlon_results.seek(0)
                csv_reader_latlon_result = csv.DictReader(latlon_results)
                for latlon_line in csv_reader_latlon_result:
                    customers.seek(0)
                    csv_reader_customers = csv.DictReader(customers)
                    lid = latlon_line["id"].strip()
                    if sid == lid:
                        for customer_line in csv_reader_customers:
                            cid = customer_line["customer_id"].strip()
                            if cid == sid == lid:
                                result[cid] = Customer(latlon_line["lat"].strip(), latlon_line["lon"].strip(),
                                                       scoring_line["value"].strip(
                                ),
                                    customer_line["input_value"].strip(
                                ),
                                    customer_line["service_time"].strip(
                                ),
                                    customer_line["visits"].strip(
                                ),
                                    customer_line["obligatory"].strip(
                                ),
                                    customer_line["opening"].strip(
                                ),
                                    customer_line["closing"].strip(
                                ),
                                    customer_line["week_day_regularity"].strip(
                                ),
                                    scoring_line["abc"].strip())
                                break
                            else:
                                pass
                        break
                    else:
                        pass
        return result

    def _get_sales_person_result_dict(self):
        result = {}
        with open(self.input()["routing_phase"]["lat_lon_result"].path, "r")as latlon_results, \
                open(self.input()["gather_and_integrate_phase"]["sales_person"].path, "r") as sales_person:

            csv_reader_latlon_result = csv.DictReader(latlon_results)

            for latlon_line in csv_reader_latlon_result:
                lid = latlon_line["id"].strip()
                sales_person.seek(0)
                csv_reader_sales_person = csv.DictReader(sales_person)

                for sales_person_line in csv_reader_sales_person:
                    sid = sales_person_line["sales_person_id"].strip()
                    if lid == sid:
                        result[sid] = Sales_Person(latlon_line["lat"].strip(), latlon_line["lon"].strip(),
                                                   sales_person_line["earliest"].strip(
                        ),
                            sales_person_line["latest"].strip(
                        ),
                            sales_person_line["time_limit"].strip(
                        ),
                            sales_person_line["days_per_week"].strip(
                        ),
                            sales_person_line["nb_weeks"].strip())
                        break
                    else:
                        pass

        return result

    def _create_mptop_instance(self, customers_dict, sales_person_dict):
        dima_result_name = basename(
            self.input()["routing_phase"]["dima_result"].path)
        customers = ""
        for i, (customer_id, customer) in enumerate(customers_dict.items()):
            if i < len(customers_dict.items()) and not i == 0:
                comma = ",\n"
            else:
                comma = ""
            customers = customers + comma + customer_node.substitute(customer_id=str(customer_id),
                                                                     customer_lon=str(
                                                                         customer.lon),
                                                                     customer_lat=str(
                                                                         customer.lat),
                                                                     customer_scoring_value=str(
                                                                         customer.scoring_value),
                                                                     customer_input_value=str(
                                                                         customer.input_value),
                                                                     customer_service_time=str(
                                                                         customer.service_time),
                                                                     customer_visits=str(
                                                                         customer.visits),
                                                                     customer_obligatory=str('true'
                                                                                             if customer.obligatory == 1
                                                                                             else 'false'),
                                                                     customer_opening=str(
                                                                         customer.opening),
                                                                     customer_closing=str(
                                                                         customer.closing),
                                                                     customer_week_day_regularity=str(
                customer.week_day_regularity),
                customer_category_abc=str(customer.category_abc))

        result = ""
        for (sales_person_id, sales_person) in sales_person_dict.items():
            result = body.substitute(nb_weeks=str(sales_person.nb_weeks),
                                     dima_result_name=str(dima_result_name),
                                     days_per_week=str(
                                         sales_person.days_per_week),
                                     sales_person_id=str(sales_person_id),
                                     sales_person_lon=str(sales_person.lon),
                                     sales_person_lat=str(sales_person.lat),
                                     sales_person_earliest=str(
                                         sales_person.earliest),
                                     sales_person_latest=str(
                                         sales_person.latest),
                                     sales_person_time_limit=str(
                                         sales_person.time_limit),
                                     customer_nodes=str(customers))

        instance_file = open(pjoin(str(self.solver_instances_result_path),
                             self._get_variant_label() + "-" + "mptop_instance.json"), "w")
        instance_file.write(result)
        instance_file_path = instance_file.name
        instance_file.close()
        return instance_file_path

    def _run_solver(self, instance_file_path):

        system("/mptop/MPTOPApp/MPTOPApp" + " " + abspath(self.input()["config"].path) + " " + abspath(instance_file_path) + " " + abspath(dirname(self.input()[
               "routing_phase"]["dima_result"].path)) + " " + abspath(self.output()["solver_result"].path) + " " + abspath(self.output()["mptop_log"].path) + " " + str(self.seed))
        solver_result_file = open(self.output()["solver_result"].path, "r")
        solver_result_file_path = solver_result_file.name
        solver_result_file.close()
        return solver_result_file_path


class CreateHashMapResult(CLSTask, globalConfig):
    abstract = False
    mptop = ClsParameter(tpe=MptopSolver.return_type())

    def requires(self):
        return self.mptop()

    def output(self):
        return {"hash_map_result": luigi.LocalTarget(pjoin(str(self.hash_map_result_path), "hash_map_result.txt")), "solver_result": self.input()["solver_result"], "hash_map_result_pickle": luigi.LocalTarget(pjoin(str(self.hash_map_result_path), "hash_map_result.pkl"))}

    def run(self):
        self._create_hash_map_txt_file()
        self._create_hash_map_pkl_file()

    def _create_hash_map_txt_file(self):
        with open(self.output()["hash_map_result"].path, "w") as hash_map_file:
            for key, value in self.hash_map.items():
                hash_map_file.write(str(key) + ": \n")
                hash_map_file.write("################################### \n")
                hash_map_file.write(str(value) + "\n")
                hash_map_file.write("=================================== \n")

    def _create_hash_map_pkl_file(self):
        with open(self.output()["hash_map_result_pickle"].path, "wb") as hash_map_pkl_file:
            pickle.dump(CLSBaseTask.hash_map, hash_map_pkl_file)

class FinalTask(CLSWrapperTask):
    abstract = False
    substitute_name = ClsParameter(tpe={1: CreateHashMapResult.return_type(),
                                        "2": MptopSolver.return_type()})
    config_domain = {1, "2"}
    def requires(self):
        return self.substitute_name()

class FinalTask_old(luigi.WrapperTask, LuigiCombinator):
    substitute_name = ClsParameter(tpe={1: CreateHashMapResult.return_type(),
                                        "2": MptopSolver.return_type()})
    config_domain = {1, "2"}
    def requires(self):
        return self.substitute_name()

def run_main():
   
    target = CreateHashMapResult.return_type()
    repository = RepoMeta.filtered_repository(
        [
        (AbstractScoringPhase,[SabcScoringPhase, NsScoringPhase]),
        DistanceMatrixAiRoutingPhase,
        MptopSolver
        ]
                                              )
    
    # repository = RepoMeta.filtered_repository(
    #     [
    #     NsScoringPhase,
    #     NSConfig1,
    #     NSConfig2,
    #     DistanceMatrixAiRoutingPhase,
    #     MptopSolver
    #     ]
    #                                           )
    
        
    # repository = RepoMeta.filtered_repository(
    #     [
    #     NsScoringPhase,
    #     (AbstractNSConfig, [NSConfig1, NSConfig2]),
    #     DistanceMatrixAiRoutingPhase,
    #     MptopSolver
    #     ]
    #                                           )
    
    #bug to fix 
    # repository = RepoMeta.filtered_repository(
    #     [
    #         SabcScoringPhase,
    #         NsScoringPhase
    #     ]
    #                                           )
    
    for item in repository:
        print("#################")
        print("key: ", str(item), " :-> ", "value: ", str(repository[item]))
        print(repository[item])
        print("#################")
        
    for item in RepoMeta.subtypes:
        print("++++++++++++++++++++")
        print("key: ", str(item), " :-> ", "value: ", str(RepoMeta.subtypes[item]))
        print("++++++++++++++++++++")
    
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
    inhabitation_result = fcl.inhabit(target)
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite
    if not actual is None or actual == 0:
        max_results = actual
    validator = RepoMeta.get_unique_abstract_task_validator()
    results = [t() for t in inhabitation_result.evaluated[0:max_results]
               if True] #validator.validate(t())
    if results:
        print("Number of results", max_results)
        print("Number of results after filtering", len(results))
        print("Run Pipelines")
        #no_schedule_error = luigi.build(results, local_scheduler=False, detailed_summary=True)
    else:
        print("No results!")


if __name__ == '__main__':
    run_main()