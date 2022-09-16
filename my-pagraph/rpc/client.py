import xmlrpc.client
server = xmlrpc.client.ServerProxy("http://localhost:8000")


server.add(1,2)

# Run 1 server and 2 clients
# As the first client blocks the server, the second is never answered.
# This is proof that using a RPCserver data communication is serailized
# Both during the transfer as there is a single socket
# And during handling as there is only one worker.
# A simple test, the time for each of the workers would be different.
# This is an over sell.
