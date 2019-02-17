#include <iostream>
#include <iomanip>
#include <fstream>
#include <conio.h>
using namespace std;


void main()
{
	char name[256], pwd[256], buf[256];
	cout << "Enter network name: ";
	cin >> name;
	cout << "Enter network password: ";
	cin >> pwd;
	sprintf_s(buf, "netsh wlan set hostednetwork mode=allow ssid=%s key=%s", name, pwd);
	system(buf);
	system("netsh wlan start hostednetwork");
	system("pause");
}


