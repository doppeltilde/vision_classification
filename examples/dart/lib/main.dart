import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

Map<String, String> headers = {
  "Content-Type": 'multipart/form-data',
  "Accept": 'application/json',
  "X-API-Key": "example",
};

Map<String, dynamic> queryParamters = {
  "label": "high",
  "score_threshold": "0.1",
  "every_n_frame": "3",
};

String queryString = Uri(queryParameters: queryParamters).query;

var url = Uri.parse('http://localhost:8000/api/classify?$queryString');

Future<void> main() async {
  final v = await fetchNetworkImage();

  var request = http.MultipartRequest("POST", url)
    ..headers.addAll(headers)
    ..files.add(http.MultipartFile.fromBytes('file', v, filename: "example"));

  final response = await http.Response.fromStream(await request.send());

  if (response.statusCode == 200) {
    print("==============================================");
    beautify(response.body);
    print("==============================================");
  }
}

Future<Uint8List> fetchNetworkImage() async {
  final examplePic = Uri.parse(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/Great_Pyrenees_Mountain_Dog_2.png/1024px-Great_Pyrenees_Mountain_Dog_2.png",
  );
  print(url);

  final response = await http.get(examplePic);

  return response.bodyBytes;
}

beautify(jsonString) {
  try {
    dynamic decodedJson = jsonDecode(jsonString);

    JsonEncoder encoder = JsonEncoder.withIndent('  ');

    String prettyPrintedJson = encoder.convert(decodedJson);

    print("\n\n$prettyPrintedJson\n\n");
  } catch (e) {
    print("Error parsing JSON: $e");
    print("\n\n$jsonString\n\n");
  }
}
